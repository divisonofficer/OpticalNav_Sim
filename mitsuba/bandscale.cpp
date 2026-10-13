#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _texture-bandscale:

Per-band scale (:monosp:`bandscale`)
------------------------------------

.. pluginparameters::

 * - nested
   - |spectrum| or |texture|
   - The value it scales, an emitter's radiance (read as an emissive spectrum unless ``emissive`` is
     false, when it is a reflectance).
 * - rgb, nir
   - |float| or |texture|
   - Factors for the visible and the near-infrared band. Shared ``uniform`` textures let one parameter
     set a whole class of lights (LED, incandescent) at once. (Default: 1 and 0)
 * - band
   - |float| or |texture|
   - Above 0.5: every lookup is near-infrared (as :monosp:`rgb2nir`'s ``band``). (Default: 0)

A lookup is near-infrared when ``band`` is set or the ray's time is 0.5 or more (a ``batch`` sensor
with ``slot_time``: its second slot), so one render holds both bands side by side, each light at its
own level in each band, without changing any parameter between them. In RGBN variants
(``cuda_rgbn_polarized``) a lookup scales R, G and B by ``rgb`` and the near-infrared lane by ``nir``.
 */
template <typename Float, typename Spectrum>
class BandScale final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    BandScale(const Properties &props) : Texture(props) {
        // an emitter's radiance: colours above 1 are allowed (an emissive spectrum, not a reflectance)
        m_nested = props.get<bool>("emissive", true) ? props.get_emissive_texture<Texture>("nested")
                                                     : props.get_texture<Texture>("nested");
        m_rgb = props.get_texture<Texture>("rgb", 1.f);
        m_nir = props.get_texture<Texture>("nir", 0.f);
        m_band = props.get_texture<Texture>("band", 0.f);
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("nested", m_nested, ParamFlags::Differentiable);
        cb->put("rgb", m_rgb, ParamFlags::NonDifferentiable);
        cb->put("nir", m_nir, ParamFlags::NonDifferentiable);
        cb->put("band", m_band, ParamFlags::NonDifferentiable);
    }

    Float factor(const SurfaceInteraction3f &si, Mask active) const {
        Mask nir = (m_band->eval_1(si, active) > .5f) || (si.time > .5f);
        return dr::select(nir, m_nir->eval_1(si, active), m_rgb->eval_1(si, active));
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        if constexpr (is_rgbn_v<Spectrum>) {  // both bands at once: each lane at its band's level
            Float f_rgb = m_rgb->eval_1(si, active);
            return m_nested->eval(si, active) *
                   UnpolarizedSpectrum(f_rgb, f_rgb, f_rgb, m_nir->eval_1(si, active));
        } else {
            return m_nested->eval(si, active) * factor(si, active);
        }
    }

    Color3f eval_3(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_nested->eval_3(si, active) * factor(si, active);
    }

    Float eval_1(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_nested->eval_1(si, active) * factor(si, active);
    }

    Float mean() const override { return m_nested->mean(); }
    ScalarVector2i resolution() const override { return m_nested->resolution(); }
    bool is_spatially_varying() const override { return m_nested->is_spatially_varying(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "BandScale[" << std::endl
            << "  nested = " << string::indent(m_nested) << "," << std::endl
            << "  rgb = " << string::indent(m_rgb) << "," << std::endl
            << "  nir = " << string::indent(m_nir) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(BandScale)
protected:
    ref<Texture> m_nested;
    ref<Texture> m_rgb;
    ref<Texture> m_nir;
    ref<Texture> m_band;

    MI_TRAVERSE_CB(Texture, m_nested, m_rgb, m_nir, m_band)
};

MI_EXPORT_PLUGIN(BandScale)
NAMESPACE_END(mitsuba)
