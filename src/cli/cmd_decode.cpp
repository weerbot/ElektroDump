// ElektroDump — `decode <dir>` (PLAN.md D11, milestone M4).
// Offline LRIT decode. `.cadu` files (1024-byte, post-deframer) are
// derandomised, RS-corrected and demultiplexed; `.cf32`/`.cu8` IQ files run
// through the BPSK demod + FEC chain. xRIT files are written to `<dir>/xrit`.
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dirent.h>

#include "cli/cli.h"
#include "core/config.h"
#include "dsp/demod_psk.h"
#include "fec/ccsds_decoder.h"
#include "util/fs_util.h"
#include "util/string_util.h"
#include "xrit/xrit_demux.h"

namespace ed::cli {

namespace {

constexpr std::size_t kCaduBytes = 1024;

bool has_suffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
               0;
}

void collect_files(const std::string& dir, std::vector<std::string>& out) {
    DIR* handle = opendir(dir.c_str());
    if (handle == nullptr) {
        return;
    }
    std::vector<std::string> names;
    while (struct dirent* entry = readdir(handle)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        names.push_back(name);
    }
    closedir(handle);
    std::sort(names.begin(), names.end());
    for (const std::string& name : names) {
        const std::string path = fs::join(dir, name);
        if (fs::is_directory(path)) {
            collect_files(path, out);
        } else if (fs::is_regular_file(path)) {
            out.push_back(path);
        }
    }
}

struct Summary {
    uint64_t files = 0;
    uint64_t cadus = 0;
    uint64_t xrit_files = 0;
    uint64_t recovered = 0;
    uint64_t uncorrectable = 0;
    bool used_iq = false;  // true once an IQ file runs the demod + FEC chain
};

void write_files(const std::string& out_dir,
                 std::vector<ed::xrit::XritFile>& files, Summary& summary) {
    for (const ed::xrit::XritFile& file : files) {
        std::string name = file.filename;
        if (name.empty()) {
            name = "vcid" + std::to_string(file.vcid) + "_apid" +
                   std::to_string(file.apid) + ".lrit";
        }
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "%06llu_",
                      static_cast<unsigned long long>(summary.xrit_files));
        const std::string path = fs::join(out_dir, std::string(prefix) + name);
        if (fs::write_file(path,
                           std::string(file.data.begin(), file.data.end()))) {
            ++summary.xrit_files;
            if (file.crc_recovered) {
                ++summary.recovered;
            }
            std::cout << "  " << path << " (" << file.data.size() << " B"
                      << (file.crc_recovered ? ", crc-recovered" : "") << ")\n";
        } else {
            std::cerr << "warning: cannot write " << path << "\n";
        }
    }
}

void decode_cadu_file(const std::string& path, fec::CcsdsDecoder& decoder,
                      ed::xrit::XritDemux& demux, const std::string& out_dir,
                      Summary& summary) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "warning: cannot open " << path << "\n";
        return;
    }
    ++summary.files;
    std::vector<uint8_t> cadu(kCaduBytes);
    std::vector<ed::xrit::XritFile> files;
    while (in.read(reinterpret_cast<char*>(cadu.data()),
                   static_cast<std::streamsize>(kCaduBytes))) {
        ++summary.cadus;
        if (!decoder.correct_cadu(cadu.data())) {
            ++summary.uncorrectable;
        }
        files.clear();
        demux.push_cadu(cadu.data(), cadu.size(), files);
        write_files(out_dir, files, summary);
    }
}

void decode_iq_file(const std::string& path, bool is_cu8, const char* section,
                    const Config& cfg, fec::CcsdsDecoder& decoder,
                    ed::xrit::XritDemux& demux, const std::string& out_dir,
                    Summary& summary) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "warning: cannot open " << path << "\n";
        return;
    }
    ++summary.files;
    summary.used_iq = true;

    const std::string sec(section);
    const bool hrit = sec == "hrit";
    dsp::DemodConfig dcfg;
    dcfg.constellation = dsp::constellation_from_string(
        cfg.get_string(sec + ".constellation", hrit ? "oqpsk" : "bpsk"));
    dcfg.symbol_rate =
        cfg.get_double(sec + ".symbol_rate", hrit ? 1150000.0 : 294000.0);
    dcfg.rrc_alpha = cfg.get_double(sec + ".rrc_alpha", 0.5);
    dcfg.pll_bw = cfg.get_double(sec + ".pll_bw", hrit ? 0.001 : 0.0145);
    // These two were live-only, so `decode` silently ran a different demod than
    // the daemon: a bandwidth sweep through this command changed nothing.
    dcfg.costas_bw = cfg.get_double(sec + ".costas_bw", dcfg.costas_bw);
    dcfg.max_sps = static_cast<int>(cfg.get_int(sec + ".max_sps", dcfg.max_sps));
    dcfg.coarse_acq = cfg.get_bool(sec + ".coarse_acq", hrit);
    dcfg.coarse_acq_range_hz =
        cfg.get_double(sec + ".coarse_acq_range_hz", 100000.0);
    dcfg.coarse_acq_fft =
        static_cast<int>(cfg.get_int(sec + ".coarse_acq_fft", 16384));
    const uint32_t sample_rate = static_cast<uint32_t>(
        cfg.get_int(sec + ".sample_rate_hz", hrit ? 6000000 : 1000000));
    dsp::PskDemod demod;
    demod.configure(dcfg, sample_rate);
    std::cerr << "demod: " << dcfg.symbol_rate << " baud @ " << sample_rate
              << " sps, rrc " << dcfg.rrc_alpha << ", pll " << dcfg.pll_bw
              << ", costas " << dcfg.costas_bw << ", max_sps " << dcfg.max_sps
              << "\n";

    constexpr std::size_t kBlock = 16384;
    std::vector<std::complex<float>> iq(kBlock);
    std::vector<int8_t> soft;
    std::vector<std::vector<uint8_t>> cadus;
    std::vector<ed::xrit::XritFile> files;

    while (in) {
        std::size_t n = 0;
        if (is_cu8) {
            std::vector<uint8_t> raw(kBlock * 2);
            in.read(reinterpret_cast<char*>(raw.data()),
                    static_cast<std::streamsize>(raw.size()));
            const std::size_t got = static_cast<std::size_t>(in.gcount());
            n = got / 2;
            for (std::size_t i = 0; i < n; ++i) {
                iq[i] = std::complex<float>(
                    (static_cast<float>(raw[2 * i]) - 127.4f) / 128.0f,
                    (static_cast<float>(raw[2 * i + 1]) - 127.4f) / 128.0f);
            }
        } else {
            in.read(reinterpret_cast<char*>(iq.data()),
                    static_cast<std::streamsize>(iq.size() *
                                                 sizeof(std::complex<float>)));
            n = static_cast<std::size_t>(in.gcount()) /
                sizeof(std::complex<float>);
        }
        if (n == 0) {
            break;
        }
        soft.clear();
        demod.process(iq.data(), n, soft);
        cadus.clear();
        decoder.push_soft(soft.data(), soft.size(), cadus);
        for (const std::vector<uint8_t>& cadu : cadus) {
            ++summary.cadus;
            files.clear();
            demux.push_cadu(cadu.data(), cadu.size(), files);
            write_files(out_dir, files, summary);
        }
    }
    std::cout << "  " << path << ": SNR " << demod.snr_db() << " dB";
    if (dcfg.coarse_acq) {
        std::cout << ", carrier offset " << demod.carrier_offset_hz() << " Hz";
    }
    std::cout << "\n";
}

}  // namespace

int cmd_decode(const Options& opts, const Args& args) {
    if (args.empty()) {
        std::cerr << "error: decode requires a CADU/IQ directory\n";
        return 2;
    }
    const std::string dir = args[0];
    if (!fs::is_directory(dir)) {
        std::cerr << "error: not a directory: " << dir << "\n";
        return 2;
    }

    std::string config_error;
    Config cfg = Config::load(opts.config_path, &config_error);
    if (!config_error.empty()) {
        std::cerr << "warning: " << config_error << " (using defaults)\n";
    }

    // --mode overrides config run.mode; HRIT uses OQPSK + coarse acquisition.
    std::string mode = opts.mode;
    if (mode.empty()) {
        mode = cfg.get_string("run.mode", "lrit");
    }
    const bool hrit = (mode == "hrit");
    const char* section = hrit ? "hrit" : "lrit";

    const std::string out_dir = cfg.get_string(std::string(section) + ".output_dir",
                                               fs::join(dir, "xrit"));
    fs::mkdir_p(out_dir);

    fec::FecConfig fec_cfg;
    fec_cfg.cadu_bits = static_cast<int>(cfg.get_int("decode.cadu_bits", 8192));
    fec_cfg.derandomize = cfg.get_bool("decode.derandomize", true);
    fec_cfg.rs_interleave = static_cast<int>(cfg.get_int("decode.rs_interleave", 4));
    fec_cfg.rs_enabled = fec_cfg.rs_interleave > 0;
    fec_cfg.rs_usecheck = cfg.get_bool("decode.rs_usecheck", true);
    fec_cfg.rs_dual_basis = cfg.get_bool("decode.rs_dualbasis", false);
    fec_cfg.viterbi_ber_threshold =
        cfg.get_double("decode.viterbi_ber_threshold", 0.30);
    fec_cfg.viterbi_outsync_after =
        static_cast<int>(cfg.get_int("decode.viterbi_outsync_after", 20));

    fec::CcsdsDecoder decoder;
    decoder.configure(fec_cfg);
    ed::xrit::XritDemux demux;

    Summary summary;
    std::vector<std::string> files;
    collect_files(dir, files);
    for (const std::string& path : files) {
        const std::string name = fs::basename(path);
        if (has_suffix(name, ".cadu")) {
            decode_cadu_file(path, decoder, demux, out_dir, summary);
        } else if (has_suffix(name, ".cf32")) {
            decode_iq_file(path, false, section, cfg, decoder, demux, out_dir,
                           summary);
        } else if (has_suffix(name, ".cu8")) {
            decode_iq_file(path, true, section, cfg, decoder, demux, out_dir,
                           summary);
        }
    }

    const fec::DecoderStats ds = decoder.stats();
    const ed::xrit::DemuxStats xs = demux.stats();
    // Label reflects the path taken: LRIT unless HRIT was selected.
    const char* label = hrit ? "hrit" : "lrit";
    std::cout << "decode (" << label << "): " << summary.files << " file(s), "
              << summary.cadus << " CADU, " << summary.xrit_files
              << " xRIT file(s)";
    if (summary.recovered > 0) {
        std::cout << " (" << summary.recovered << " crc-recovered)";
    }
    std::cout << "\n  ";
    // Viterbi/deframer only run for IQ input; CADUs are already deframed.
    if (summary.used_iq) {
        std::cout << "viterbi BER " << ds.ber << ", viterbi "
                  << (ds.viterbi_locked ? "locked" : "unlocked") << ", deframer "
                  << (ds.deframer_locked ? "locked" : "unlocked") << ", ";
    }
    std::cout << "RS corrected " << ds.rs_corrected << ", uncorrectable "
              << ds.cadus_uncorrectable << "\n  demux: " << xs.files
              << " file(s), vcdu filler " << xs.vcdu_filler << "\noutput: "
              << out_dir << "\n";
    return 0;
}

}  // namespace ed::cli
