// Minimal Spike extension model for ToyRoCC custom0 instructions.

#include <array>
#include <riscv/mmu.h>
#include <riscv/rocc.h>


class toyrocc_t : public rocc_t {
 public:
  toyrocc_t() { regfile.fill(0); }
  const char* name() const override { return "toy_rocc"; }

  // accumulator example
  reg_t custom0(processor_t* p, rocc_insn_t insn, reg_t xs1, reg_t xs2) override {
    const size_t addr = static_cast<size_t>(xs2) & (kNumRegs - 1);
    const reg_t prior = regfile[addr];

    switch (insn.funct) {
      case 0:  // write: reg[addr] = xs1
        regfile[addr] = xs1;
        break;
      case 1:  // read: return reg[addr]
        break;
      case 2:  // load: reg[addr] = *(uint64_t*)xs1
        regfile[addr] = p->get_mmu()->load<uint64_t>(xs1);
        break;
      case 3:  // accum: reg[addr] += xs1
        regfile[addr] = prior + xs1;
        break;
      default:
        // Keep behavior minimal: unknown funct acts as a no-op.
        break;
    }

    // RoCC response semantics in the RTL return prior accumulator value.
    return prior;
  }

  // vector exp
  reg_t custom1(processor_t* p, rocc_insn_t insn, reg_t xs1, reg_t xs2) override {
    const size_t addr = static_cast<size_t>(xs2) & (kNumRegs - 1);
    const reg_t prior = regfile[addr];

    switch (insn.funct) {
      case 0:  // write: reg[addr] = xs1
        regfile[addr] = xs1;
        break;
      case 1:  // read: return reg[addr]
        break;
      case 2:  // load: reg[addr] = *(uint64_t*)xs1
        regfile[addr] = p->get_mmu()->load<uint64_t>(xs1);
        break;
      case 3:  // accum: reg[addr] += xs1
        regfile[addr] = prior + xs1;
        break;
      default:
        // Keep behavior minimal: unknown funct acts as a no-op.
        break;
    }

    // RoCC response semantics in the RTL return prior accumulator value.
    return prior;
  }

 private:
  static constexpr size_t kNumRegs = 4;
  std::array<reg_t, kNumRegs> regfile;
};

REGISTER_EXTENSION(toy_rocc, []() { return new toyrocc_t; })
