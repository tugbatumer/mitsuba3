#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-fluorescent:
*/
template <typename Float, typename Spectrum>
class Fluorescent final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    SmoothDiffuse(const Properties &props) : Base(props) {
        m_reflectance = props.texture<Texture>("reflectance", .5f);
        m_flags = BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide;
        m_components.push_back(m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("reflectance", m_reflectance.get(), +ParamFlags::Differentiable);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /* sample1 */,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs = dr::zeros<BSDFSample3f>();

        active &= cos_theta_i > 0.f;
        if (unlikely(dr::none_or<false>(active) ||
                     !ctx.is_enabled(BSDFFlags::DiffuseReflection)))
            return { bs, 0.f };

        bs.wo = warp::square_to_cosine_hemisphere(sample2);
        bs.pdf = warp::square_to_cosine_hemisphere_pdf(bs.wo);
        bs.eta = 1.f;
        bs.sampled_type = +BSDFFlags::DiffuseReflection;
        bs.sampled_component = 0;

        UnpolarizedSpectrum value = m_reflectance->eval(si, active);

        return { bs, depolarizer<Spectrum>(value) & (active && bs.pdf > 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::DiffuseReflection))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        UnpolarizedSpectrum value =
            m_reflectance->eval(si, active) * dr::InvPi<Float> * cos_theta_o;

        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::DiffuseReflection))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

        return dr::select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf, 0.f);
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (!ctx.is_enabled(BSDFFlags::DiffuseReflection))
            return { 0.f, 0.f };

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        UnpolarizedSpectrum value =
            m_reflectance->eval(si, active) * dr::InvPi<Float> * cos_theta_o;

        Float pdf = warp::square_to_cosine_hemisphere_pdf(wo);

        return { depolarizer<Spectrum>(value) & active, dr::select(active, pdf, 0.f) };
    }

    Spectrum eval_diffuse_reflectance(const SurfaceInteraction3f &si,
                                      Mask active) const override {
        return m_reflectance->eval(si, active);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SmoothDiffuse[" << std::endl
            << "  reflectance = " << string::indent(m_reflectance) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Texture> m_reflectance;
};

MI_IMPLEMENT_CLASS_VARIANT(SmoothDiffuse, BSDF)
MI_EXPORT_PLUGIN(SmoothDiffuse, "Smooth diffuse material")
NAMESPACE_END(mitsuba)
