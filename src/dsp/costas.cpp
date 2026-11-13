#include "dsp/costas.h"

#include <cmath>

namespace ed::dsp {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

double clip(double value, double limit) {
    if (value > limit) {
        return limit;
    }
    if (value < -limit) {
        return -limit;
    }
    return value;
}

}  // namespace

void CostasLoop::configure(double loop_bw, int order) {
    if (!(loop_bw > 0.0)) {
        loop_bw = 0.02;
    }
    order_ = (order == 4) ? 4 : 2;

    const double damping = std::sqrt(2.0) / 2.0;
    const double denom = 1.0 + 2.0 * damping * loop_bw + loop_bw * loop_bw;
    alpha_ = (4.0 * damping * loop_bw) / denom;
    beta_ = (4.0 * loop_bw * loop_bw) / denom;
    reset();
}

void CostasLoop::reset() {
    phase_ = 0.0;
    freq_ = 0.0;
    power_ = 0.0;
}

void CostasLoop::process(const std::complex<float>* in, std::size_t n,
                         std::complex<float>* out) {
    if (in == nullptr || out == nullptr) {
        return;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const double c = std::cos(phase_);
        const double s = std::sin(phase_);
        const double re = static_cast<double>(in[i].real());
        const double im = static_cast<double>(in[i].imag());
        const double mixed_re = re * c + im * s;
        const double mixed_im = im * c - re * s;

        // Track the signal power so the loop gain does not depend on the front
        // end's amplitude. A real L-band chain delivers ~0.03 after the RRC,
        // which made alpha/beta effectively 30x too small: the loop held at
        // 12 dB (residual error small) and lost lock at 5 dB (phase noise large).
        const double inst_pwr = re * re + im * im;
        power_ = (power_ > 0.0) ? power_ + 0.002 * (inst_pwr - power_)
                                : inst_pwr;
        out[i] = std::complex<float>(static_cast<float>(mixed_re),
                                     static_cast<float>(mixed_im));

        double error = 0.0;
        if (order_ == 2) {
            // BPSK: the detector must use the *decision* (sign of the in-phase
            // rail), i.e. sign(I)*Q. Using I*Q makes the error flip sign with
            // every symbol, so it averages to zero and the loop never locks: the
            // constellation just circles. (QPSK's four-quadrant detector below
            // is decision-directed already, which is why only BPSK was broken.)
            // SatDump's BPSK detector is I*Q, not sign(I)*Q: with a loop
            // bandwidth well below the symbol rate the per-symbol sign flip
            // averages out, and it keeps the linear S-curve near lock where
            // sign() is a hard nonlinearity. This only works when the input is
            // AGC'd to unit amplitude, which demod_psk now does.
            error = mixed_re * mixed_im;
        } else {
            const double sign_re = (mixed_re > 0.0) ? 1.0 : -1.0;
            const double sign_im = (mixed_im > 0.0) ? 1.0 : -1.0;
            error = sign_re * mixed_im - sign_im * mixed_re;
        }
        // Normalise to unit amplitude before the loop filter.
        const double norm = 1.0 / (std::sqrt(power_) + 1e-9);
        error = clip(error * norm, 1.0);

        freq_ += beta_ * error;
        phase_ += freq_ + alpha_ * error;

        if (!std::isfinite(phase_) || !std::isfinite(freq_)) {
            phase_ = 0.0;
            freq_ = 0.0;
        }
        while (phase_ > kTwoPi) {
            phase_ -= kTwoPi;
        }
        while (phase_ < -kTwoPi) {
            phase_ += kTwoPi;
        }
        // Keep the tracked frequency within a sane range.
        freq_ = clip(freq_, kPi / 4.0);
    }
}

}  // namespace ed::dsp
