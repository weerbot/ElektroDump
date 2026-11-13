// ElektroDump — CCSDS CADU deframer (PLAN.md D3.5, digest §4.4).
// Bit-level search for the ASM 0x1ACFFC1D and its inverse. Syncing requires a
// run of good ASMs before lock; Synced drops straight back to NoSync on a bad
// ASM so a polarity inversion can be re-detected. Emits 1024-byte CADUs with the
// ASM in normal polarity at bytes 0..3.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ed::fec {

class CcsdsDeframer {
public:
    enum class State { NoSync, Syncing, Synced };

    // cadu_bits defaults to 8192 (1024 bytes). asm is the sync word.
    void configure(std::size_t cadu_bits = 8192, uint32_t asm_word = 0x1ACFFC1D);
    void reset();

    // Tolerant ASM comparison: allow up to this many bit errors (0 or 1).
    void set_max_bit_errors(int errors) { max_errors_ = errors; }

    // Push hard bits (0/1); append each complete CADU to `out`.
    void push(const uint8_t* bits, std::size_t n,
              std::vector<std::vector<uint8_t>>& out);

    State state() const { return state_; }
    bool locked() const { return state_ == State::Synced; }
    std::size_t cadu_bytes() const { return cadu_bytes_; }
    uint64_t frames() const { return frames_; }

private:
    void begin_frame();
    void write_body_bit(uint8_t bit);
    void finish_frame(std::vector<std::vector<uint8_t>>& out);
    void on_next_asm(bool good, std::vector<std::vector<uint8_t>>& out);
    static int bit_errors(uint32_t a, uint32_t b);

    std::size_t cadu_bits_ = 8192;
    std::size_t cadu_bytes_ = 1024;
    uint32_t asm_ = 0x1ACFFC1D;
    uint32_t asm_inv_ = 0xE53003E2;
    // CCSDS 131.0 allows the ASM to be acquired with up to 2 bit errors. At 0
    // an exact 32-bit match is required, which at low SNR rejects about half the
    // frames and drops the deframer out of lock; nothing ever set this.
    int max_errors_ = 2;

    State state_ = State::NoSync;
    bool inversion_ = false;
    uint32_t shifter_ = 0;

    bool collecting_ = false;
    std::size_t bit_of_frame_ = 0;  // bits written including the 32-bit ASM
    std::vector<uint8_t> frame_;

    bool expecting_asm_ = false;
    uint32_t asm_acc_ = 0;
    int asm_count_ = 0;
    int good_asm_ = 0;
    int bad_asm_ = 0;
    uint64_t frames_ = 0;
};

}  // namespace ed::fec
