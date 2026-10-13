#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _texture-rgb2nir:

RGB or pseudo near-infrared reflectance (:monosp:`rgb2nir`)
-----------------------------------------------------------

.. pluginparameters::

 * - nested
   - |spectrum| or |texture|
   - The visible (linear RGB) reflectance it wraps, typically a base color.
 * - low, high
   - |float| or |texture|
   - The pseudo near-infrared reflectance spans 0.5 (mid grey) to 1 (black or white) by construction
     (``max(c, 1 - c) >= 0.5``); it is remapped linearly onto [``low``, ``high``]:
     ``low + (high - low) * (n - 0.5) / 0.5``. Like ``band``, give every wrapper a reference to shared
     ``uniform`` textures to set a whole scene's range at render time. (Default: 0.5 and 1, unchanged)
 * - band
   - |float| or |texture|
   - 0: return ``nested`` unchanged. 1: return the pseudo near-infrared reflectance
     ``dot(max(c, 1 - c), [0.229, 0.587, 0.114])`` of ``c = clip(nested, 0, 1)`` in every channel
     (robomituba's ``pseudo_nir_albedo``, used for its Infinigen, HSSD and ABO corpora).
     Give every wrapper a reference to one shared ``uniform`` texture, so that one parameter switches
     a whole scene between the visible and the near-infrared band. (Default: 0)
     A ray whose time is 0.5 or more (a ``batch`` sensor with ``slot_time``: its second slot) also gets
     the near-infrared reflectance, so one render can hold both bands side by side. In RGBN variants
     (``cuda_rgbn_polarized``) the lookup returns both at once: R, G, B and the near-infrared lane.

The switch is a value read at render time, so a frozen render serves both bands without recording
again, and no second set of textures is loaded: the near-infrared reflectance is computed per lookup.
 */
template <typename Float, typename Spectrum>
class RGBToNIR final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    RGBToNIR(const Properties &props) : Texture(props) {
        m_nested = props.get_texture<Texture>("nested");
        m_band = props.get_texture<Texture>("band", 0.f);
        m_low = props.get_texture<Texture>("low", .5f);
        m_high = props.get_texture<Texture>("high", 1.f);
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("nested", m_nested, ParamFlags::Differentiable);
        cb->put("band", m_band, ParamFlags::NonDifferentiable);
        cb->put("low", m_low, ParamFlags::NonDifferentiable);
        cb->put("high", m_high, ParamFlags::NonDifferentiable);
    }

    /// pseudo_nir (0.5 .. 1) remapped onto [low, high]
    Float nir_of(const Color3f &rgb, const SurfaceInteraction3f &si, Mask active) const {
        Float lo = m_low->eval_1(si, active), hi = m_high->eval_1(si, active);
        return dr::fmadd(hi - lo, (pseudo_nir(rgb) - .5f) * 2.f, lo);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        UnpolarizedSpectrum value = m_nested->eval(si, active);
        if constexpr (is_rgbn_v<Spectrum>) {  // both bands at once: the near-infrared lane
            Color3f rgb = rgb_lanes(value);
            return UnpolarizedSpectrum(rgb[0], rgb[1], rgb[2], nir_of(rgb, si, active));
        } else if constexpr (is_rgb_v<Spectrum>) {
            Mask nir = (m_band->eval_1(si, active) > .5f) || (si.time > .5f);
            Float n = nir_of(Color3f(value), si, active);  // mitsuba::pseudo_nir (spectrum.h), remapped
            return dr::select(nir, UnpolarizedSpectrum(n), value);
        } else {
            return value;  // spectral and monochrome variants: unchanged
        }
    }

    Color3f eval_3(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        Color3f value = m_nested->eval_3(si, active);
        Mask nir = (m_band->eval_1(si, active) > .5f) || (si.time > .5f);
        return dr::select(nir, Color3f(nir_of(value, si, active)), value);
    }

    Float eval_1(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        Mask nir = (m_band->eval_1(si, active) > .5f) || (si.time > .5f);
        return dr::select(nir, nir_of(m_nested->eval_3(si, active), si, active), m_nested->eval_1(si, active));
    }

    Float mean() const override { return m_nested->mean(); }
    ScalarVector2i resolution() const override { return m_nested->resolution(); }
    bool is_spatially_varying() const override { return m_nested->is_spatially_varying(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "RGBToNIR[" << std::endl
            << "  nested = " << string::indent(m_nested) << "," << std::endl
            << "  band = " << string::indent(m_band) << "," << std::endl
            << "  low = " << string::indent(m_low) << "," << std::endl
            << "  high = " << string::indent(m_high) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(RGBToNIR)
protected:
    ref<Texture> m_nested;
    ref<Texture> m_band;
    ref<Texture> m_low;
    ref<Texture> m_high;

    MI_TRAVERSE_CB(Texture, m_nested, m_band, m_low, m_high)
};

MI_EXPORT_PLUGIN(RGBToNIR)
NAMESPACE_END(mitsuba)
