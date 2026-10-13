#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path_nocaustics:

Path tracer without caustics (:monosp:`path_nocaustics`)
--------------------------------------------------------

Identical to :monosp:`path`, except that emitter contributions reaching the
light through a *near-specular* bounce are discarded once the path has already
scattered off a rough surface. This is the "caustics off" option of production
renderers: diffuse -> mirror/glass -> small light paths cannot be sampled
efficiently by a unidirectional path tracer and appear as fireflies, whereas
camera-side specular chains (seeing through glass, seeing a lit wall in a
mirror, seeing the light's reflection) are kept.

A bounce is near-specular when its sampled lobe is delta/null, or when the
BSDF pdf of the direction exceeds ``specular_pdf_threshold`` (sr^-1). The same
classification is applied to emitter sampling and to BSDF-sampled emitter hits,
so both MIS strategies drop the same set of paths.

.. pluginparameters::

 * - max_depth, rr_depth, hide_emitters
   - as :monosp:`path`
 * - specular_pdf_threshold
   - |float|
   - Solid-angle pdf above which a bounce counts as near-specular. The GGX peak
     pdf is about 1/(4 pi alpha^2), so the default 20 covers alpha below ~0.06.
     (Default: 20)
 * - caustics
   - |bool|
   - Keep caustic paths, so the integrator behaves exactly like :monosp:`path`
     (apart from the camera-assist options below). (Default: false)
 * - assist, assist_mode
   - |string|
   - Camera-assist shapes are the shapes whose id starts with ``assist`` (for
     example ``camera_assist`` matches ``camera_assist_light`` and
     ``camera_assist_polarizer``). With ``assist_mode`` = ``exclude`` the path
     ignores them: rays pass through them without a bounce and their emitter
     is never counted, as if they were not in the scene. With ``only``, the
     emitter on an assist shape is the only light: other emitters (including
     the environment) are ignored, and it is sampled directly, as if it were
     the scene's single emitter. ``all`` (the default) changes nothing. This
     lets one resident scene render a passive pass and a flash-only pass.
 * - sky_nir
   - |float|
   - Factor on light from the environment emitter in the near-infrared: its lane in RGBN variants,
     otherwise along rays whose time is 0.5 or more (the near-infrared slot of a ``batch`` sensor with
     ``slot_time``). Daylight's near-infrared share.
     (Default: 1)
 */

template <typename Float, typename Spectrum>
class PathNoCausticsIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr, Shape, ShapePtr)

    enum class AssistMode { All, Only, Exclude };

    PathNoCausticsIntegrator(const Properties &props) : Base(props) {
        m_specular_pdf_threshold = props.get<ScalarFloat>("specular_pdf_threshold", 20.f);
        m_caustics = props.get<bool>("caustics", false);
        m_sky_nir = props.get<ScalarFloat>("sky_nir", 1.f);
        m_assist = std::string(props.get<std::string_view>("assist", ""));
        std::string mode(props.get<std::string_view>("assist_mode", "all"));
        if (mode == "all")
            m_assist_mode = AssistMode::All;
        else if (mode == "only")
            m_assist_mode = AssistMode::Only;
        else if (mode == "exclude")
            m_assist_mode = AssistMode::Exclude;
        else
            Throw("path_nocaustics: 'assist_mode' must be all, only or exclude (got \"%s\")", mode);
        if (m_assist_mode != AssistMode::All && m_assist.empty())
            Throw("path_nocaustics: 'assist_mode' %s needs an 'assist' shape id prefix", mode);
    }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        PreliminaryIntersection3f pi  = dr::zeros<PreliminaryIntersection3f>();
        UInt32 depth                  = 0;

        // Camera-assist shapes (see 'assist'): a few scalar compares per call
        std::vector<const Shape *> assist_shapes;
        const Emitter *assist_emitter = nullptr;
        if (m_assist_mode != AssistMode::All) {
            for (const auto &shape : scene->shapes()) {
                if (!string::starts_with(shape->id(), m_assist))
                    continue;
                assist_shapes.push_back(shape.get());
                if (shape->emitter())
                    assist_emitter = shape->emitter();
            }
            if (m_assist_mode == AssistMode::Only && !assist_emitter)
                Throw("path_nocaustics: no emitter on a shape whose id starts with \"%s\"", m_assist);
        }

        auto is_assist = [&assist_shapes](const ShapePtr &shape) {
            Mask hit = false;
            for (const Shape *s : assist_shapes)
                hit |= (shape == ShapePtr(s));
            return hit;
        };

        // 'exclude': continue the ray through assist shapes as if they were absent (no bounce is counted).
        // Each one is flat, so a ray crosses it at most once.
        auto skip_assist = [&](Ray3f &ray, PreliminaryIntersection3f &pi, Mask active) {
            for (size_t i = 0; i < assist_shapes.size(); ++i) {
                Mask hit = active && pi.is_valid() && is_assist(pi.shape);
                if (dr::none_or<false>(hit))
                    break;
                SurfaceInteraction3f si = pi.compute_surface_interaction(ray, +RayFlags::Minimal, hit);
                Ray3f next = si.spawn_ray(ray.d);
                PreliminaryIntersection3f pi_next =
                    scene->ray_intersect_preliminary(next, false, false, 0, 0, hit);
                dr::masked(ray, hit) = next;
                dr::masked(pi, hit) = pi_next;
            }
        };

        // If m_hide_emitters == false, the environment emitter will be visible
        // ('only' ignores every emitter but the assist one, the environment included)
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr) &&
                         (m_assist_mode != AssistMode::Only);

        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.
        */
        struct LoopState {
            Ray3f ray;
            PreliminaryIntersection3f pi;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Bool rough_seen;
            Bool prev_caustic;
            Bool active;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, ray, pi, throughput, result, eta, depth, \
                valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                rough_seen, prev_caustic, active, sampler)
        } ls = {
            ray,
            pi,
            throughput,
            result,
            eta,
            depth,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            /* rough_seen   = */ Bool(false),
            /* prev_caustic = */ Bool(false),
            active,
            sampler
        };

        // First bounce is usually coherent - don't reorder threads
        ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                 /* coherent = */ true,
                                                 /* reorder = */ false,
                                                 /* reorder_hint = */ 0,
                                                 /* reorder_hint_bits = */ 0,
                                                 ls.active);
        if (m_assist_mode == AssistMode::Exclude)
            skip_assist(ls.ray, ls.pi, ls.active);

        // ---------------------- Hide area emitters ----------------------

        /* dr::any_or() checks for active entries in the provided boolean
           array. JIT/Megakernel modes can't do this test efficiently as
           each Monte Carlo sample runs independently. In this case,
           dr::any_or<..>() returns the template argument (true) which means
           that the 'if' statement is always conservatively taken. */

        if (m_hide_emitters && dr::any_or<true>(ls.depth == 0u)) {
            // Did we hit an area emitter? If so, skip all area emitters along this ray
            Mask skip_emitters = ls.pi.is_valid() &&
                                 (ls.pi.shape->emitter() != nullptr) &&
                                 ls.active;

            if (dr::any_or<true>(skip_emitters)) {
                SurfaceInteraction3f si = ls.pi.compute_surface_interaction(
                    ls.ray, +RayFlags::Minimal, skip_emitters);
                Ray3f ray = si.spawn_ray(ls.ray.d);
                PreliminaryIntersection3f pi_after_skip =
                    Base::skip_area_emitters(scene, ray, true, skip_emitters);
                dr::masked(ls.pi, skip_emitters) = pi_after_skip;
            }
        }

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, bsdf_ctx, &skip_assist, assist_emitter](LoopState& ls) {

            /* dr::while_loop implicitly masks all code in the loop using the
               'active' flag, so there is no need to pass it to every function */

            // Fill out all information of the interaction
            SurfaceInteraction3f si =
                ls.pi.compute_surface_interaction(ls.ray, +RayFlags::All);

            // ---------------------- Direct emission ----------------------

            if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, si, ls.prev_si);
                Float em_pdf = 0.f;

                // 'only': the assist emitter alone counts, with its own pdf (no emitter-selection pmf)
                Mask counted = true;
                if (m_assist_mode == AssistMode::Only)
                    counted = ds.emitter == EmitterPtr(assist_emitter);

                if (dr::any_or<true>(!ls.prev_bsdf_delta)) {
                    if (m_assist_mode == AssistMode::Only)
                        em_pdf = assist_emitter->pdf_direction(ls.prev_si, ds,
                                                               !ls.prev_bsdf_delta && counted);
                    else
                        em_pdf = scene->pdf_emitter_direction(ls.prev_si, ds,
                                                              !ls.prev_bsdf_delta);
                }

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);

                // Accumulate, being careful with polarization (see spec_fma)
                // Reached through a near-specular bounce after a rough one: a caustic.
                Mask hit_active = (ls.prev_bsdf_pdf > 0.f) && counted;
                if (!m_caustics)
                    hit_active &= !ls.prev_caustic;
                // the environment: daylight's near-infrared share in the near-infrared lane (RGBN) or slot
                ls.result = spec_fma(
                    ls.throughput,
                    sky_nir(ds.emitter->eval(si, hit_active) * mis_bsdf, !si.is_valid(), ls.ray.time),
                    ls.result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next)) {
                ls.active = active_next;
                ls.valid_ray |= (si.emitter(scene) != nullptr) && !m_hide_emitters;
                return; // early exit for scalar mode
            }

            BSDFPtr bsdf = si.bsdf(ls.ray);

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                if (m_assist_mode == AssistMode::Only) {
                    // as Scene::sample_emitter_direction would with the assist emitter as its only one
                    std::tie(ds, em_weight) = assist_emitter->sample_direction(
                        si, ls.sampler->next_2d(), active_em);
                    active_em &= (ds.pdf != 0.f);
                    if (dr::any_or<true>(active_em)) {
                        Mask occluded = scene->ray_test(si.spawn_ray_to(ds.p), active_em);
                        dr::masked(em_weight, occluded) = 0.f;
                        dr::masked(ds.pdf, occluded) = 0.f;
                    }
                } else {
                    std::tie(ds, em_weight) = scene->sample_emitter_direction(
                        si, ls.sampler->next_2d(), true, active_em);
                    if (m_assist_mode == AssistMode::Exclude)
                        active_em &= ds.emitter != EmitterPtr(assist_emitter);
                    if (m_sky_nir != 1.f && scene->environment() != nullptr)
                        em_weight = sky_nir(em_weight, ds.emitter == EmitterPtr(scene->environment()), si.time);
                }
                active_em &= (ds.pdf != 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(ds.pdf != 0, em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1 = ls.sampler->next_1d();
            Point2f sample_2 = ls.sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

            // --------------- Emitter sampling contribution ----------------

            // Emitter sample through a near-specular lobe after a rough bounce.
            if (!m_caustics)
                active_em &= !(ls.rough_seen && (bsdf_pdf > m_specular_pdf_threshold));

            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                ls.result[active_em] = spec_fma(
                    ls.throughput, bsdf_val * em_weight * mis_em, ls.result);
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ls.ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ls.ray)) {
                ls.ray = dr::detach<true>(ls.ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ls.ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2, ls.active);
                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            ls.throughput *= bsdf_weight;
            ls.eta *= bsdf_sample.eta;
            ls.valid_ray |= ls.active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            ls.prev_si = Interaction3f(si);
            ls.prev_bsdf_pdf = bsdf_sample.pdf;
            ls.prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            Mask near_specular =
                has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta) ||
                has_flag(bsdf_sample.sampled_type, BSDFFlags::Null) ||
                (bsdf_sample.pdf > m_specular_pdf_threshold);
            ls.prev_caustic = near_specular && ls.rough_seen;
            ls.rough_seen |= !near_specular;

            // -------------------- Stopping criterion ---------------------

            dr::masked(ls.depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
            Mask rr_active = ls.depth >= m_rr_depth,
                 rr_continue = ls.sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            ls.active = active_next && (!rr_active || rr_continue) &&
                        (throughput_max != 0.f);

            // Reorder threads based on the shape they hit
            ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                     /* coherent = */ false,
                                                     /* reorder = */ jit_flag(JitFlag::LoopRecord),
                                                     /* reorder_hint = */ 0,
                                                     /* reorder_hint_bits = */ 0,
                                                     ls.active);
            if (m_assist_mode == AssistMode::Exclude)
                skip_assist(ls.ray, ls.pi, ls.active);
        });

        return {
            /* spec  = */ dr::select(ls.valid_ray, ls.result, 0.f),
            /* valid = */ ls.valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathNoCausticsIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u,\n"
            "  specular_pdf_threshold = %f,\n"
            "  caustics = %s,\n"
            "  assist = \"%s\" (%s)\n"
            "]", m_max_depth, m_rr_depth, m_specular_pdf_threshold, m_caustics,
            m_assist, m_assist_mode == AssistMode::Only ? "only"
                      : m_assist_mode == AssistMode::Exclude ? "exclude" : "all");
    }

    /**
     * \brief Light from the environment (where ``sky``) at daylight's near-infrared share: in RGBN
     * variants on the near-infrared lane, otherwise on rays of a dual render's near-infrared slot (time
     * 0.5 or more).
     */
    Spectrum sky_nir(const Spectrum &value, Mask sky, Float time) const {
        if (m_sky_nir == 1.f)
            return value;
        if constexpr (is_rgbn_v<Spectrum>) {
            UnpolarizedSpectrum f(1.f, 1.f, 1.f, dr::select(sky, m_sky_nir, 1.f));
            Spectrum out = value;
            if constexpr (is_polarized_v<Spectrum>) {
                for (size_t i = 0; i < 4; ++i)
                    for (size_t j = 0; j < 4; ++j)
                        out.entry(i, j) *= f;
            } else {
                out *= f;
            }
            return out;
        } else {
            return value * dr::select(sky && (time > .5f), m_sky_nir, 1.f);
        }
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    ScalarFloat m_specular_pdf_threshold;
    bool m_caustics;
    ScalarFloat m_sky_nir;
    std::string m_assist;
    AssistMode m_assist_mode;

    MI_DECLARE_CLASS(PathNoCausticsIntegrator)
};

MI_EXPORT_PLUGIN(PathNoCausticsIntegrator)
NAMESPACE_END(mitsuba)
