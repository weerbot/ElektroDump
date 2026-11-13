#include "dsp/demod_psk.h"

#include <cstdio>
#include <cstdlib>

#include <cmath>
#include <mutex>

namespace ed::dsp {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

int8_t to_soft(double value) {
    if (value > 127.0) {
        return 127;
    }
    if (value < -127.0) {
        return -127;
    }
    return static_cast<int8_t>(std::lround(value));
}

}  // namespace

Constellation constellation_from_string(const std::string& name) {
    if (name == "qpsk") {
        return Constellation::QPSK;
    }
    if (name == "oqpsk") {
        return Constellation::OQPSK;
    }
    return Constellation::BPSK;
}

void PskDemod::configure(const DemodConfig& cfg, uint32_t sample_rate) {
    cfg_ = cfg;
    sample_rate_ = sample_rate;
    if (!(cfg_.symbol_rate > 0.0)) {
        cfg_.symbol_rate = 294000.0;
    }
    if (cfg_.rrc_taps < 3) {
        cfg_.rrc_taps = 33;
    }

    // A wideband capture delivers far more samples per symbol than the
    // demodulator wants (10 MSPS on a 294 kbaud LRIT carrier is sps 34, and the
    // RRC filter alone then costs more than a Pi 4 core). Decimate to the
    // configured samples/symbol first, through the anti-alias FIR below.
    const int target_sps = cfg_.max_sps >= 2 ? cfg_.max_sps : 3;
    if (cfg_.symbol_rate > 0.0) {
        const double want = static_cast<double>(sample_rate_) /
                            (static_cast<double>(target_sps) * cfg_.symbol_rate);
        decim_ = static_cast<std::size_t>(std::max(1.0, std::round(want)));
    } else {
        decim_ = 1;
    }
    dec_sum_ = std::complex<float>(0.0f, 0.0f);
    dec_count_ = 0;
    dec_rate_ = static_cast<double>(sample_rate_) / static_cast<double>(decim_);

    sps_ = dec_rate_ / cfg_.symbol_rate;
    if (!(sps_ > 1.0)) {
        sps_ = 2.0;
    }

    oqpsk_ = (cfg_.constellation == Constellation::OQPSK);
    fixed_timing_ = std::getenv("ED_FIXED_TIMING") != nullptr;
    fixed_phase_ = 0.0;
    if (fixed_timing_) {
        const char* p = std::getenv("ED_FIXED_PHASE");
        if (p != nullptr) { fixed_phase_ = std::atof(p); }
    }
    const int order = (cfg_.constellation == Constellation::BPSK) ? 2 : 4;

    // Anti-alias filter at the input rate, cutoff just under the decimated
    // Nyquist, long enough to matter (8 taps per output sample).
    {
        const std::size_t taps = 4 * decim_ + 1;
        const double fc = 0.42 / static_cast<double>(decim_);  // cycles/sample in
        aa_taps_.assign(taps, 0.0f);
        double sum = 0.0;
        const double mid = static_cast<double>(taps - 1) / 2.0;
        for (std::size_t i = 0; i < taps; ++i) {
            const double t = static_cast<double>(i) - mid;
            const double sinc = (std::fabs(t) < 1e-9) ? 1.0 : std::sin(2.0 * kPi * fc * t) / (2.0 * kPi * fc * t);
            const double w = 0.54 - 0.46 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(taps - 1));
            aa_taps_[i] = static_cast<float>(2.0 * fc * sinc * w);
            sum += aa_taps_[i];
        }
        for (float& v : aa_taps_) {
            v = static_cast<float>(v / sum);
        }
        aa_line_.assign(taps, std::complex<float>(0.0f, 0.0f));
        aa_pos_ = 0;
        aa_count_ = 0;
    }

    rrc_.configure(sps_, cfg_.rrc_alpha, cfg_.rrc_taps);
    if (oqpsk_) {
        aligner_.configure(sps_);
    }
    // The Costas is the *fine* carrier loop, running after the PLL has already
    // taken out the bulk offset. Driving it with the PLL bandwidth (14.5 kHz at
    // 1 MSPS) injects phase noise straight into the slicer: on a clean 13 dB
    // LRIT recording that alone put the input BER near 10% and stopped Viterbi
    // from ever holding. Separate, much narrower default.
    costas_.configure(cfg_.costas_bw, order);
    evm_.configure(order, 0.002);
    // OQPSK symbol instants sit half a symbol later (I rail delayed by T/2).
    const double initial_delay =
        static_cast<double>(rrc_.delay()) + (oqpsk_ ? 0.5 * sps_ : 0.0);
    clock_.configure(sps_, 0.05, 0.0005, initial_delay);

    coarse_enabled_ = cfg_.coarse_acq;
    if (coarse_enabled_) {
        coarse_.configure(static_cast<double>(sample_rate_),
                          cfg_.coarse_acq_range_hz,
                          static_cast<std::size_t>(cfg_.coarse_acq_fft), order);
    }
    acquired_ = false;
    dec_sum_ = std::complex<float>(0.0f, 0.0f);
    dec_count_ = 0;
    corr_freq_hz_ = 0.0;
    corr_phase_ = 0.0;
    acq_buf_.clear();

    rms_ = 0.0;
    m2m4_.reset();
    snr_db_.store(0.0, std::memory_order_relaxed);
    carrier_offset_hz_.store(0.0, std::memory_order_relaxed);
    locked_.store(false, std::memory_order_relaxed);
}

void PskDemod::reset() {
    rrc_.reset();
    if (oqpsk_) {
        aligner_.reset();
    }
    costas_.reset();
    clock_.reset();
    coarse_.reset();
    acquired_ = false;
    corr_freq_hz_ = 0.0;
    corr_phase_ = 0.0;
    acq_buf_.clear();
    rms_ = 0.0;
    agc_gain_ = 1.0f;
    if (sym_dump_ == nullptr) {
        const char* path = std::getenv("ED_DUMP_PSK_SYMS");
        if (path != nullptr) {
            sym_dump_ = std::fopen(path, "wb");
        }
    }
    evm_.reset();
    m2m4_.reset();
    snr_db_.store(0.0, std::memory_order_relaxed);
    carrier_offset_hz_.store(0.0, std::memory_order_relaxed);
    locked_.store(false, std::memory_order_relaxed);
}

void PskDemod::push_soft(double value, std::vector<int8_t>& soft_out) {
    const double scale = 100.0 / std::sqrt(rms_ + 1e-9);
    soft_out.push_back(to_soft(value * scale));
    if (sym_dump_ != nullptr) {
        const double v = value * scale;
        const long q = static_cast<long>(v < -127.0 ? -127.0 : (v > 127.0 ? 127.0 : v));
        std::fputc(static_cast<int>(q) + 128, sym_dump_);
    }
}

void PskDemod::process(const std::complex<float>* iq, std::size_t n,
                       std::vector<int8_t>& soft_out) {
    if (iq == nullptr || n == 0) {
        return;
    }

    // Buffer a full FFT block, estimate the carrier offset once, then process
    // the buffered samples with the correction applied.
    if (coarse_enabled_ && !acquired_) {
        acq_buf_.insert(acq_buf_.end(), iq, iq + n);
        const std::size_t need = static_cast<std::size_t>(cfg_.coarse_acq_fft);
        if (acq_buf_.size() < need) {
            return;
        }
        corr_freq_hz_ = coarse_.estimate(acq_buf_.data(), acq_buf_.size());
        carrier_offset_hz_.store(corr_freq_hz_, std::memory_order_relaxed);
        acquired_ = true;
        process_block(acq_buf_.data(), acq_buf_.size(), soft_out);
        acq_buf_.clear();
        acq_buf_.shrink_to_fit();
        return;
    }

    process_block(iq, n, soft_out);
}

void PskDemod::process_block(const std::complex<float>* iq, std::size_t n,
                             std::vector<int8_t>& soft_out) {
    const std::complex<float>* src = iq;
    // Wideband capture: shift this mode's carrier down to DC, on top of any
    // coarse-acquisition correction.
    // The coarse estimate is measured on the raw stream, so it already contains
    // the wideband offset: adding both would double-count it.
    const double static_offset_hz =
        (coarse_enabled_ && acquired_) ? corr_freq_hz_ : cfg_.freq_offset_hz;
    if (static_offset_hz != 0.0) {
        corr_buf_.resize(n);
        // Rotation is exp(-j*corr_phase_); a positive offset advances phase.
        // Recursive rotation: one sin/cos per block instead of two per sample,
        // which is the difference between 10 MSPS fitting and not on a Pi.
        const double dphi =
            kTwoPi * static_offset_hz / static_cast<double>(sample_rate_);
        std::complex<float> rot(static_cast<float>(std::cos(corr_phase_)),
                                static_cast<float>(-std::sin(corr_phase_)));
        const std::complex<float> delta(static_cast<float>(std::cos(dphi)),
                                        static_cast<float>(-std::sin(dphi)));
        for (std::size_t i = 0; i < n; ++i) {
            corr_buf_[i] = iq[i] * rot;
            rot *= delta;
            if ((i & 0x3FFu) == 0) {
                const float mag = std::abs(rot);
                if (mag > 1e-6f) {
                    rot /= mag;
                } else {
                    rot = std::complex<float>(1.0f, 0.0f);
                }
            }
        }
        corr_phase_ += dphi * static_cast<double>(n);
        while (corr_phase_ > kPi) {
            corr_phase_ -= kTwoPi;
        }
        while (corr_phase_ < -kPi) {
            corr_phase_ += kTwoPi;
        }
        src = corr_buf_.data();
    }

    if (decim_ > 1) {
        dec_buf_.clear();
        dec_buf_.reserve(n / decim_ + 2);
        const std::size_t taps = aa_taps_.size();
        for (std::size_t i = 0; i < n; ++i) {
            aa_line_[aa_pos_] = src[i];
            if (++aa_pos_ >= taps) {
                aa_pos_ = 0;
            }
            if (++dec_count_ < decim_) {
                continue;
            }
            // One FIR output per decim_ inputs. The taps are symmetric, so the
            // traversal start does not matter.
            std::complex<float> acc(0.0f, 0.0f);
            std::size_t idx = aa_pos_;
            for (std::size_t k = 0; k < taps; ++k) {
                acc += aa_taps_[k] * aa_line_[idx];
                if (++idx >= taps) {
                    idx = 0;
                }
            }
            dec_buf_.push_back(acc);
            dec_count_ = 0;
        }
        src = dec_buf_.data();
        n = dec_buf_.size();
    }

    rrc_buf_.resize(n);
    rrc_.process(src, n, rrc_buf_.data());

    // No AGC: it normalises the amplitude to 1.0, which multiplies the effective
    // Costas and timing-loop gains by the signal amplitude (about 30x on a real
    // capture) and destabilises loops that are tuned for the raw level.

    // Carrier recovery must precede OQPSK alignment: the aligner delays only
    // the real part by T/2, valid only once derotated. A residual carrier makes
    // the rails cross-couple and inverts the Gardner timing S-curve.
    // Debug hook: dump the demapped symbols for comparison against a reference
    // demodulator on the same capture.
    // AGC to the unit circle before the carrier loop, as SatDump does. Without
    // it every loop gain scales with the front end's amplitude (about 0.03
    // here), which is why the loops only held on strong passes.
    if (agc_enabled_) {
        agc_buf_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::complex<float> out = rrc_buf_[i] * agc_gain_;
            const float mag = std::abs(out);
            agc_gain_ += kAgcRate * (1.0f - mag);
            if (agc_gain_ > 65536.0f) { agc_gain_ = 65536.0f; }
            if (agc_gain_ < 0.0f) { agc_gain_ = 0.0f; }
            agc_buf_[i] = out;
        }
        costas_buf_.resize(n);
        costas_.process(agc_buf_.data(), n, costas_buf_.data());
    } else {
        costas_buf_.resize(n);
        costas_.process(rrc_buf_.data(), n, costas_buf_.data());
    }

    // OQPSK: align the I rail with the Q rail (turns it into QPSK).
    const std::complex<float>* aligned = costas_buf_.data();
    if (oqpsk_) {
        align_buf_.resize(n);
        aligner_.process(costas_buf_.data(), n, align_buf_.data());
        aligned = align_buf_.data();
    }

    symbols_.clear();
    if (fixed_timing_) {
        // Diagnostic only: sample at a constant spacing, carrying the phase
        // across blocks. Valid when the symbol clock is exactly known.
        if (fixed_phase_ <= 0.0) { fixed_phase_ = sps_ * 0.5; }
        symbols_.clear();
        symbols_.reserve(static_cast<std::size_t>(n / sps_) + 2);
        while (fixed_phase_ < static_cast<double>(n)) {
            const std::size_t idx = static_cast<std::size_t>(fixed_phase_);
            symbols_.push_back(aligned[idx]);
            fixed_phase_ += sps_;
        }
        fixed_phase_ -= static_cast<double>(n);
    } else {
        clock_.process(aligned, n, symbols_);
    }
    if (symbols_.empty()) {
        return;
    }

    evm_.update(symbols_.data(), symbols_.size());

    {
        constexpr std::size_t kKeep = 8192;
        std::lock_guard<std::mutex> lock(sym_mutex_);
        if (sym_ring_.size() != kKeep) {
            sym_ring_.assign(kKeep, std::complex<float>(0.0f, 0.0f));
            sym_pos_ = 0;
            sym_count_ = 0;
        }
        for (const std::complex<float>& sym : symbols_) {
            sym_ring_[sym_pos_] = sym;
            sym_pos_ = (sym_pos_ + 1) % kKeep;
            if (sym_count_ < kKeep) {
                ++sym_count_;
            }
        }
    }
    m2m4_.update(src, n);

    // BPSK emits the I rail; OQPSK emits I then Q, the order Viterbi consumes.
    for (const std::complex<float>& sym : symbols_) {
        const double re = static_cast<double>(sym.real());
        const double im = static_cast<double>(sym.imag());
        const double power = re * re + (oqpsk_ ? im * im : 0.0);
        rms_ = 0.99 * rms_ + 0.01 * power;
        push_soft(re, soft_out);
        if (oqpsk_) {
            push_soft(im, soft_out);
        }
    }

    const double evm_snr = evm_.snr_db();
    const double m2m4_snr = m2m4_.snr_db();
    const double snr = (evm_.samples() >= 64) ? evm_snr : m2m4_snr;
    snr_db_.store(snr, std::memory_order_relaxed);
    locked_.store(snr > 3.0 && evm_.samples() >= 64,
                  std::memory_order_relaxed);
}

void PskDemod::constellation(std::vector<std::complex<float>>& out) const {
    std::lock_guard<std::mutex> lock(sym_mutex_);
    out.clear();
    if (sym_count_ == 0) {
        return;
    }
    const std::size_t cap = sym_ring_.size();
    const std::size_t start = (sym_pos_ + cap - sym_count_) % cap;
    out.reserve(sym_count_);
    for (std::size_t i = 0; i < sym_count_; ++i) {
        out.push_back(sym_ring_[(start + i) % cap]);
    }
}

double PskDemod::snr_db() const {
    return snr_db_.load(std::memory_order_relaxed);
}

bool PskDemod::locked() const {
    return locked_.load(std::memory_order_relaxed);
}

}  // namespace ed::dsp
