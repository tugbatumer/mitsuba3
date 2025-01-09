#pragma once

#include <mitsuba/core/logger.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/math.h>
#include <drjit/dynamic.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/util.h>
#include <array>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Discrete 2D wavelength distribution
 *
 * This data structure represents a discrete 1D probability distribution and
 * provides various routines for transforming uniformly distributed samples so
 * that they follow the stored distribution. Note that unnormalized
 * probability mass functions (PMFs) will automatically be normalized during
 * initialization. The associated scale factor can be retrieved using the
 * function \ref normalization().
 */
template <typename Float_, size_t Dimension_ = 0>
class WavelengthDistribution {
public:
    using Float                       = Float_;
    using UInt32                      = dr::uint32_array_t<Float>;
    using Mask                        = dr::mask_t<Float>;
    using Point2f                     = Point<Float, 2>;
    using Point2u                     = Point<UInt32, 2>;
    using ScalarFloat                 = dr::scalar_t<Float>;
    using ScalarVector2u              = Vector<uint32_t, 2>;
    using FloatStorage                = DynamicBuffer<Float>;

    WavelengthDistribution() = default;

    /**
     * Construct a marginal sample scheme for floating point
     * data of resolution \c size.
     * Assumes that rows are incoming and columns are outgoing wavelengths.
     */
    WavelengthDistribution(const ScalarFloat *data,
                           const ScalarVector2u &size)
        : m_size(size) {

        std::unique_ptr<ScalarFloat[]> cond_cdf(new ScalarFloat[dr::prod(m_size)]);
        std::unique_ptr<ScalarFloat[]> marg_cdf(new ScalarFloat[m_size.y()]);

        // Construct conditional and marginal CDFs
        double accum_marg = 0.0;
        for (uint32_t y = 0; y < m_size.y(); ++y) {
            double accum_cond = 0.0;
            uint32_t idx = m_size.x() * y;
            for (uint32_t x = 0; x < m_size.x(); ++x, ++idx) {
                accum_cond += (double) data[idx];
                cond_cdf[idx] = (ScalarFloat) accum_cond;
            }
            accum_marg += accum_cond;
            marg_cdf[y] = (ScalarFloat) accum_marg;
        }

        m_cond_cdf = dr::load<FloatStorage>(cond_cdf.get(), dr::prod(m_size));
        m_marg_cdf = dr::load<FloatStorage>(marg_cdf.get(), m_size.y());

        m_inv_normalization = dr::opaque<Float>(accum_marg);
        m_normalization = dr::opaque<Float>(1.0 / accum_marg);
    }

    Float eval_pdf(const UInt32 &index, Mask active, Mask zero) const {
        return dr::gather<Float>(m_cond_cdf, index, active) -
               dr::gather<Float>(m_cond_cdf, index - 1, active && zero);
    }

    /// Evaluate the function value at the given float wavelengths
    Float eval(const Point2f &pos, Mask active = true) const {
        UInt32 row_index = floor((pos.x() - 300) / 10);
        UInt32 column_index = floor((pos.y() - 380) / 10);

        UInt32 top_left = m_size.x() * row_index + column_index;
        UInt32 top_right = m_size.x() * row_index + column_index + 1;
        UInt32 bottom_left = m_size.x() * (row_index + 1) + column_index;
        UInt32 bottom_right = m_size.x() * (row_index + 1) + column_index + 1;

        Float incoming_mod = pos.x() - floor(pos.x() / 10.f) * 10.f;
        Float outgoing_mod = pos.y() - floor(pos.y() / 10.f) * 10.f;

        return (
        (10 - incoming_mod) *
        (
            (10 - outgoing_mod) * eval_pdf(top_left, active, column_index != 0) +
            outgoing_mod * eval_pdf(top_right, active, true)
        ) / 10 +
        incoming_mod *
        (
            (10 - outgoing_mod) * eval_pdf(bottom_left, active, column_index != 0) +
            outgoing_mod * eval_pdf(bottom_right, active, true)
        ) / 10)
        / 100;
    }

    /// Evaluate the normalized function value at the given integer position
    Float pdf(const Point2u &pos, Mask active = true) const {
        return eval(pos, active) * m_normalization;
    }

    /**
     * \brief Given a uniformly distributed 2D sample, draw a sample from the
     * distribution
     *
     * Returns the sampled wavelength and the normalized probability value
     */
    std::tuple<Float, Float> sample(const Point2f &sample_,
                                               Mask active = true) const {
        MI_MASK_ARGUMENT(active);
        Float incoming = sample_.x();
        Float sample = sample_.y();

        UInt32 row_index = floor((incoming - 300) / 10);
        Float incoming_mod = incoming - floor(incoming / 10.f) * 10.f;

        UInt32 top = m_size.x() * row_index;
        UInt32 bottom = m_size.x() * (row_index + 1);

        // Sample the index from the interpolated cdf
        UInt32 index = dr::binary_search<UInt32>(
            0u, m_size.x() - 1, [&](UInt32 idx) DRJIT_INLINE_LAMBDA {
                return (((10 - incoming_mod) * dr::gather<Float>(m_cond_cdf, top + idx, active) +
                (incoming_mod) * dr::gather<Float>(m_cond_cdf, bottom + idx, active)) / 10) < sample;
            });

        Float cdf_low = (((10 - incoming_mod) * dr::gather<Float>(m_cond_cdf, top + index - 1, active && index > 0) +
                (incoming_mod) * dr::gather<Float>(m_cond_cdf, bottom + index - 1, active && index > 0)) / 10);
        Float cdf_high = (((10 - incoming_mod) * dr::gather<Float>(m_cond_cdf, top + index, active && index < m_size.x()) +
                (incoming_mod) * dr::gather<Float>(m_cond_cdf, bottom + index, active && index < m_size.x())) / 10);


        Float residual = sample - cdf_low;
        Float outgoing = 380 + ((index + residual / (cdf_high - cdf_low)) * 10);
        Float pdf = eval(Point2f(incoming, outgoing));
        return {380 + ((index + residual / (cdf_high - cdf_low)) * 10), pdf};
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "WavelengthDistribution" << "[" << std::endl
            << "  size = " << m_size << "," << std::endl
            << "  normalization = " << m_normalization << std::endl
            << "]";
        return oss.str();
    }

protected:
    /// Resolution of the discretized density function
    ScalarVector2u m_size;

    /// Density values
    FloatStorage m_data;

    /// Marginal and conditional PDFs
    FloatStorage m_marg_cdf;
    FloatStorage m_cond_cdf;

    Float m_inv_normalization;
    Float m_normalization;
};

NAMESPACE_END(mitsuba)
