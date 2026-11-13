// ElektroDump — Costas carrier/phase recovery loop (PLAN.md D2/D3).
// Decision-directed PLL for BPSK (order 2) and QPSK (order 4); runs after the
// matched filter and before symbol timing recovery.
#pragma once

#include <complex>
#include <cstddef>

namespace ed::dsp {

class CostasLoop {
public:
    void configure(double loop_bw, int order = 2);
    void reset();

    // `out` may alias `in`.
    void process(const std::complex<float>* in, std::size_t n,
                 std::complex<float>* out);

    double phase() const { return phase_; }
    double freq() const { return freq_; }

private:
    double alpha_ = 0.0;
    double beta_ = 0.0;
    double phase_ = 0.0;
    double freq_ = 0.0;
    double power_ = 0.0;
    int order_ = 2;
};

}  // namespace ed::dsp
