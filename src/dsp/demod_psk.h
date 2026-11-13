// ElektroDump — PSK demodulator (PLAN.md D3.2/D3.3).
// Chain: IQ -> coarse carrier acq -> RRC -> Costas -> OQPSK align -> Gardner
// -> int8 soft symbols. BPSK (LRIT) emits one soft symbol per symbol; OQPSK
// (HRIT) aligns the rails and emits I then Q, the order the rate-1/2 Viterbi
// decoder consumes.
#pragma once

#include <atomic>
#include <mutex>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <cstdio>
#include <vector>

#include "dsp/coarse_acq.h"
#include "dsp/costas.h"
#include "dsp/gardner_timing.h"
#include "dsp/oqpsk_align.h"
#include "dsp/rrc_filter.h"
#include "dsp/snr_estimator.h"

namespace ed::dsp {

enum class Constellation { BPSK, QPSK, OQPSK };

struct DemodConfig {
    Constellation constellation = Constellation::BPSK;
    double symbol_rate = 294000.0;
    double rrc_alpha = 0.5;
    double pll_bw = 0.0145;    // bulk carrier loop (coarse)
    double costas_bw = 0.004;  // fine carrier loop, much narrower
    int max_sps = 3;
    int rrc_taps = 33;

    // Digital offset between the radio centre and this mode's carrier.
    double freq_offset_hz = 0.0;

    // When enabled the demod buffers the first `coarse_acq_fft` samples,
    // estimates the offset and removes it before the Costas loop.
    bool coarse_acq = false;
    double coarse_acq_range_hz = 100000.0;
    int coarse_acq_fft = 16384;
};

class PskDemod {
public:
    void configure(const DemodConfig& cfg, uint32_t sample_rate);
    void reset();

    // Consumes IQ and appends int8 soft symbols (range -127..127).
    void process(const std::complex<float>* iq, std::size_t n,
                 std::vector<int8_t>& soft_out);

    double snr_db() const;
    bool locked() const;
    double symbol_rate() const { return cfg_.symbol_rate; }
    // Most recent symbols for the constellation plot (thread-safe copy).
    void constellation(std::vector<std::complex<float>>& out) const;
    double samples_per_symbol() const { return sps_; }

    // Coarse carrier estimate in Hz (0 until acquisition has run).
    double carrier_offset_hz() const {
        return carrier_offset_hz_.load(std::memory_order_relaxed);
    }
    bool acquired() const { return acquired_; }

private:
    void process_block(const std::complex<float>* iq, std::size_t n,
                       std::vector<int8_t>& soft_out);
    void push_soft(double value, std::vector<int8_t>& soft_out);

    DemodConfig cfg_;
    uint32_t sample_rate_ = 0;
    double sps_ = 1.0;
    bool oqpsk_ = false;

    RrcFilter rrc_;
    OqpskAligner aligner_;
    CostasLoop costas_;
    GardnerTiming clock_;
    EvmSnrEstimator evm_{2, 0.002};
    M2m4SnrEstimator m2m4_{0.001};

    CoarseAcquisition coarse_;
    bool coarse_enabled_ = false;
    bool acquired_ = false;
    double corr_freq_hz_ = 0.0;
    double corr_phase_ = 0.0;
    std::vector<std::complex<float>> acq_buf_;

    double rms_ = 0.0;
    FILE* sym_dump_ = nullptr;
    std::vector<std::complex<float>> corr_buf_;
    std::vector<std::complex<float>> rrc_buf_;
    std::vector<std::complex<float>> align_buf_;
    std::vector<std::complex<float>> costas_buf_;
    std::vector<std::complex<float>> symbols_;

    mutable std::mutex sym_mutex_;
    std::vector<std::complex<float>> sym_ring_;
    std::size_t sym_pos_ = 0;
    std::size_t sym_count_ = 0;

    // Input decimation to the configured samples/symbol.
    std::size_t decim_ = 1;
    // Anti-alias FIR run at the input rate before subsampling. A boxcar folds
    // every Nyquist zone into the band (about 3 dB of noise for decim 2, more
    // for 3) and adds ISI, which is enough to keep RS from correcting.
    std::vector<float> aa_taps_;
    std::vector<std::complex<float>> aa_line_;
    std::size_t aa_pos_ = 0;
    std::size_t aa_count_ = 0;
    // AGC feeding the carrier loop, as SatDump does: normalise to unit
    // amplitude so the loop gains mean the same thing at every input level.
    static constexpr float kAgcRate = 1e-4f;
    bool agc_enabled_ = true;
    // Diagnostic: sample at a fixed phase instead of running the timing loop.
    // Only meaningful when the true symbol clock is known exactly (synthetic).
    bool fixed_timing_ = false;
    double fixed_phase_ = 0.0;
    float agc_gain_ = 1.0f;
    std::vector<std::complex<float>> agc_buf_;
    double dec_rate_ = 0.0;
    std::complex<float> dec_sum_{0.0f, 0.0f};
    std::size_t dec_count_ = 0;
    std::vector<std::complex<float>> dec_buf_;

    std::atomic<double> snr_db_{0.0};
    std::atomic<double> carrier_offset_hz_{0.0};
    std::atomic<bool> locked_{false};
};

Constellation constellation_from_string(const std::string& name);

}  // namespace ed::dsp
