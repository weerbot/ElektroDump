#include "core/pipeline_xrit.h"

#include <chrono>
#include <complex>
#include <cstdio>
#include <thread>

#include "core/config.h"
#include "core/logging.h"
#include "core/metrics.h"
#include "core/thread_util.h"
#include "image/product_builder.h"
#include "util/fs_util.h"

namespace ed {

namespace {

constexpr std::size_t kIqBlock = 16384;

}  // namespace

XritPipeline::XritPipeline() = default;

XritPipeline::~XritPipeline() { stop(); }

bool XritPipeline::configure(const Config& cfg, const PipelineParams& params) {
    config_ = &cfg;
    params_ = params;
    mode_ = params.mode == PipelineMode::HRIT ? PipelineMode::HRIT
                                              : PipelineMode::LRIT;
    const char* section = (mode_ == PipelineMode::HRIT) ? "hrit" : "lrit";

    output_dir_ = cfg.get_string(std::string(section) + ".output_dir", "data/xrit");

    dsp::DemodConfig demod_cfg;
    demod_cfg.constellation = dsp::constellation_from_string(params.constellation);
    demod_cfg.freq_offset_hz = params.freq_offset_hz;
    demod_cfg.symbol_rate = params.symbol_rate > 0.0
                                ? params.symbol_rate
                                : cfg.get_double(std::string(section) + ".symbol_rate",
                                                 294000.0);
    demod_cfg.rrc_alpha = params.rrc_alpha > 0.0
                              ? params.rrc_alpha
                              : cfg.get_double(std::string(section) + ".rrc_alpha", 0.5);
    demod_cfg.pll_bw = params.pll_bw > 0.0
                           ? params.pll_bw
                           : cfg.get_double(std::string(section) + ".pll_bw", 0.0145);
    demod_cfg.costas_bw =
        cfg.get_double(std::string(section) + ".costas_bw", 0.004);
    demod_cfg.rrc_taps = static_cast<int>(cfg.get_int("decode.rrc_taps", 33));
    // Was never read here either, so the per-mode sample rate had no effect on
    // how far the front end decimates before the loops.
    demod_cfg.max_sps =
        static_cast<int>(cfg.get_int(std::string(section) + ".max_sps", 3));
    // Coarse carrier acquisition: on by default for HRIT, where a realistic
    // LNB/LO offset exceeds the Costas loop's pull-in range.
    demod_cfg.coarse_acq = cfg.get_bool(std::string(section) + ".coarse_acq",
                                        mode_ == PipelineMode::HRIT);
    demod_cfg.coarse_acq_range_hz =
        cfg.get_double(std::string(section) + ".coarse_acq_range_hz", 100000.0);
    demod_cfg.coarse_acq_fft = static_cast<int>(
        cfg.get_int(std::string(section) + ".coarse_acq_fft", 16384));

    const uint32_t sample_rate =
        params.sample_rate_hz > 0
            ? params.sample_rate_hz
            : static_cast<uint32_t>(
                  cfg.get_int(std::string(section) + ".sample_rate_hz", 1000000));
    demod_.configure(demod_cfg, sample_rate);

    fec::FecConfig fec_cfg;
    fec_cfg.cadu_bits = params.cadu_bits > 0 ? params.cadu_bits : 8192;
    fec_cfg.derandomize = params.derandomize;
    fec_cfg.rs_interleave = params.rs_interleave;
    fec_cfg.rs_enabled = params.rs_interleave > 0;
    fec_cfg.rs_usecheck = params.rs_usecheck;
    // CCSDS RS transmits in the dual basis; SatDump defaults this to true and real
    // LRIT recordings only decode with it on (13/120 frames had all-zero
    // syndromes with the dual-basis conversion, 0/120 without).
    fec_cfg.rs_dual_basis = cfg.get_bool("decode.rs_dualbasis", true);
    fec_cfg.viterbi_ber_threshold =
        cfg.get_double("decode.viterbi_ber_threshold", 0.30);
    fec_cfg.viterbi_outsync_after =
        static_cast<int>(cfg.get_int("decode.viterbi_outsync_after", 20));
    decoder_.configure(fec_cfg);

    demux_.reset();
    file_counter_ = 0;
    if (!fs::mkdir_p(output_dir_)) {
        ED_LOG(Warn, "lrit") << "cannot create output dir " << output_dir_;
    }

    ED_LOG(Info, "lrit")
        << "configured: " << params.constellation << " " << demod_cfg.symbol_rate
        << " baud @ " << sample_rate << " sps"
        << (demod_cfg.coarse_acq ? ", coarse-acq on" : "") << ", output "
        << output_dir_;
    return true;
}

void XritPipeline::attach_io(const PipelineIo& io) { io_ = io; }

void XritPipeline::start() {
    if (running_.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    last_stats_ = std::chrono::steady_clock::now();
    thread_ = std::thread([this] { dsp_loop(); });
}

void XritPipeline::stop() {
    running_.store(false, std::memory_order_relaxed);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void XritPipeline::write_file(const xrit::XritFile& file) {
    std::string name = file.filename;
    if (name.empty()) {
        name = "vcid" + std::to_string(file.vcid) + "_apid" +
               std::to_string(file.apid) + ".lrit";
    }
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "%06llu_",
                  static_cast<unsigned long long>(file_counter_++));
    const std::string path = fs::join(output_dir_, std::string(prefix) + name);
    if (!fs::write_file(path, std::string(file.data.begin(), file.data.end()))) {
        ED_LOG(Warn, "lrit") << "failed to write " << path;
        return;
    }
    ED_LOG(Info, "lrit") << "assembled " << file.data.size() << " B -> " << path
                         << (file.crc_recovered ? " (crc-recovered)" : "");
    if (io_.products != nullptr) {
        io_.products->submit(file);
    }
}

void XritPipeline::dsp_loop() {
    thread::set_name("lrit-dsp");
    thread::set_nice(-10);

    std::vector<std::complex<float>> iq(kIqBlock);
    std::vector<int8_t> soft;
    std::vector<std::vector<uint8_t>> cadus;
    std::vector<xrit::XritFile> files;
    soft.reserve(kIqBlock);
    files.reserve(4);

    while (running_.load(std::memory_order_relaxed)) {
        if (io_.iq_ring == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        const std::size_t n = io_.iq_ring->read(iq.data(), iq.size());
        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (io_.waterfall_ring != nullptr) {
            io_.waterfall_ring->write(iq.data(), n);
        }

        soft.clear();
        demod_.process(iq.data(), n, soft);

        cadus.clear();
        decoder_.push_soft(soft.data(), soft.size(), cadus);

        for (const std::vector<uint8_t>& cadu : cadus) {
            files.clear();
            demux_.push_cadu(cadu.data(), cadu.size(), files);
            for (const xrit::XritFile& file : files) {
                write_file(file);
                if (io_.metrics != nullptr) {
                    io_.metrics->add_file(file.crc_recovered);
                }
            }
        }

        const auto now = std::chrono::steady_clock::now();
        if (now - last_stats_ >= std::chrono::seconds(1)) {
            last_stats_ = now;
            publish_stats();
        }
    }
    publish_stats();
}

void XritPipeline::publish_stats() {
    const fec::DecoderStats ds = decoder_.stats();
    const xrit::DemuxStats& xs = demux_.stats();
    // Viterbi lock alone flickers; require recent CADU activity for "locked".
    const bool locked = ds.viterbi_locked || ds.deframer_locked;

    snr_db_.store(demod_.snr_db(), std::memory_order_relaxed);
    ber_.store(ds.ber, std::memory_order_relaxed);
    locked_.store(locked, std::memory_order_relaxed);
    cadus_total_.store(ds.cadus_total, std::memory_order_relaxed);
    cadus_uncorrectable_.store(ds.cadus_uncorrectable, std::memory_order_relaxed);
    files_total_.store(xs.files, std::memory_order_relaxed);
    files_recovered_.store(xs.files_recovered, std::memory_order_relaxed);

    if (io_.metrics != nullptr) {
        io_.metrics->set_snr(demod_.snr_db());
        io_.metrics->set_ber(ds.ber);
        io_.metrics->set_locked(locked);
        io_.metrics->set_cadus(ds.cadus_total, ds.cadus_uncorrectable);
        io_.metrics->add_rs_corrected(0);
    }
}

void XritPipeline::constellation(std::vector<std::complex<float>>& out) const {
    demod_.constellation(out);
}

PipelineStats XritPipeline::stats() const {
    PipelineStats s;
    s.running = running_.load(std::memory_order_relaxed);
    s.locked = locked_.load(std::memory_order_relaxed);
    s.demod_locked = demod_.locked();
    s.ber = ber_.load(std::memory_order_relaxed);
    s.snr_db = snr_db_.load(std::memory_order_relaxed);
    s.cadus_total = cadus_total_.load(std::memory_order_relaxed);
    s.cadus_bad = cadus_uncorrectable_.load(std::memory_order_relaxed);
    s.files_total = files_total_.load(std::memory_order_relaxed);
    s.files_recovered = files_recovered_.load(std::memory_order_relaxed);
    return s;
}

}  // namespace ed
