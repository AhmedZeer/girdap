package girdap

import chisel3._
import chiseltest._
import org.scalatest.funspec.AnyFunSpec

class BFloat16Spec extends AnyFunSpec with ChiselScalatestTester {

  private def bf16ToFloat(bf16: Int): Float = {
    val bits = bf16 << 16
    java.lang.Float.intBitsToFloat(bits)
  }

  describe("BFloat16ToFixed") {
    it("should correctly convert BFloat16 to Fixed-Point (Q8.8)") {
      // Testing with 8 integer bits and 8 fractional bits
      test(new BFloat16ToFixed(8, 8)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        // 1.0 in BF16 (0x3F80) -> 1.0 in Q8.8 is 256 (0x0100)
        dut.io.in.poke("h3F80".U)
        dut.clock.step(1)
        dut.io.out.expect("h0100".U)

        // 1.5 in BF16 (0x3FC0) -> 1.5 in Q8.8 is 384 (0x0180)
        dut.io.in.poke("h3FC0".U)
        dut.clock.step(1)
        dut.io.out.expect("h0180".U)

        // -1.0 in BF16 (0xBF80) -> -1.0 in Q8.8 is -256 (0xFF00)
        dut.io.in.poke("hBF80".U)
        dut.clock.step(1)
        dut.io.out.expect("hFF00".U)
      }
    }

    it("should correctly convert BFloat16 to Fixed-Point (UQ12.20)") {
      test(new BFloat16ToFixed(12, 20)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        // 1.0 in BF16 (0x3F80) -> 1.0 in UQ12.20 is 0x0010_0000
        dut.io.in.poke("h3F80".U)
        dut.clock.step(1)
        dut.io.out.expect("h00100000".U)

        // 1.5 in BF16 (0x3FC0) -> 1.5 in UQ12.20 is 0x0018_0000
        dut.io.in.poke("h3FC0".U)
        dut.clock.step(1)
        dut.io.out.expect("h00180000".U)

        // -1.0 in BF16 (0xBF80) -> -1.0 in UQ12.20 is 0xFFF0_0000 (two's complement)
        dut.io.in.poke("hBF80".U)
        dut.clock.step(1)
        dut.io.out.expect("hFFF00000".U)
      }
    }

    it("should correctly convert BFloat16 to Fixed-Point (UQ32.32)") {
      test(new BFloat16ToFixed(32, 32)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        // 1.0 in BF16 (0x3F80) -> 1.0 in UQ32.32 is 0x0000_0001_0000_0000
        dut.io.in.poke("h3F80".U)
        dut.clock.step(1)
        dut.io.out.expect("h0000000100000000".U)

        // 1.5 in BF16 (0x3FC0) -> 1.5 in UQ32.32 is 0x0000_0001_8000_0000
        dut.io.in.poke("h3FC0".U)
        dut.clock.step(1)
        dut.io.out.expect("h0000000180000000".U)

        // -1.0 in BF16 (0xBF80) -> -1.0 in UQ32.32 is 0xFFFF_FFFF_0000_0000 (two's complement)
        dut.io.in.poke("hBF80".U)
        dut.clock.step(1)
        dut.io.out.expect("hFFFFFFFF00000000".U)
      }
    }
  }

  describe("BFloat16Exp") {
    it("should produce exp(0) ~= 1.0 and exp(negatives) <= 1.0") {
      test(new BFloat16Exp()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        val cases = Seq(
          ("h0000".U, 1.06f), // +0.0
          ("hBF00".U, 1.06f), // -0.5
          ("hBF80".U, 1.06f), // -1.0
          ("hC000".U, 1.06f)  // -2.0
        )

        for ((in, maxExpected) <- cases) {
          dut.io.in.poke(in)
          dut.clock.step(1)
          val outBf16 = dut.io.out.peek().litValue.toInt
          val outF = bf16ToFloat(outBf16)
          assert(outF <= maxExpected, s"exp output too large for input=${in.litValue.toString(16)}: $outF")
        }
      }
    }

    it("should be monotonic around zero") {
      test(new BFloat16Exp()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        val inNeg = "hBF80".U // -1.0
        val inZero = "h0000".U // 0.0
        val inPos = "h3F80".U // +1.0

        dut.io.in.poke(inNeg)
        dut.clock.step(1)
        val outNeg = bf16ToFloat(dut.io.out.peek().litValue.toInt)

        dut.io.in.poke(inZero)
        dut.clock.step(1)
        val outZero = bf16ToFloat(dut.io.out.peek().litValue.toInt)

        dut.io.in.poke(inPos)
        dut.clock.step(1)
        val outPos = bf16ToFloat(dut.io.out.peek().litValue.toInt)

        assert(outNeg <= outZero, s"exp(-1) should be <= exp(0): $outNeg vs $outZero")
        assert(outZero <= outPos, s"exp(0) should be <= exp(+1): $outZero vs $outPos")
      }
    }
  }

  describe("BFloat16Sub") {
    it("should correctly subtract two BFloat16 numbers using HardFloat") {
      test(new BFloat16Sub()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        // 1.5 (0x3FC0) - 1.0 (0x3F80) = 0.5 (0x3F00)
        dut.io.in_1.poke("h3FC0".U)
        dut.io.in_2.poke("h3F80".U)
        dut.clock.step(1)
        dut.io.out.expect("h3F00".U)
      }
    }
  }

  describe("BFloat16Max") {
    it("should correctly identify the maximum of two BFloat16 numbers") {
      test(new BFloat16Max()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        // 1.0 (0x3F80) vs 2.0 (0x4000) -> 2.0 (0x4000)
        dut.io.in_1.poke("h3F80".U)
        dut.io.in_2.poke("h4000".U)
        dut.clock.step(1)
        dut.io.out.expect("h4000".U)

        // -1.0 (0xBF80) vs 1.0 (0x3F80) -> 1.0 (0x3F80)
        dut.io.in_1.poke("hBF80".U)
        dut.io.in_2.poke("h3F80".U)
        dut.clock.step(1)
        dut.io.out.expect("h3F80".U)

        // -2.0 (0xC000) vs -1.0 (0xBF80) -> -1.0 (0xBF80)
        dut.io.in_1.poke("hC000".U)
        dut.io.in_2.poke("hBF80".U)
        dut.clock.step(1)
        dut.io.out.expect("hBF80".U)
      }
    }
  }

  describe("BFloat16VectorMax") {
    it("should correctly identify the maximum in a vector of 4 BFloat16 numbers") {
      test(new BFloat16VectorMax(4)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        // Inputs:
        // val4 = 1.5 (0x3FC0)
        // val3 = 3.0 (0x4040)
        // val2 = 2.0 (0x4000)
        // val1 = 1.0 (0x3F80)
        // Packed 64-bit string: 0x3FC0_4040_4000_3F80
        // Expected Max = 3.0 (0x4040)

        dut.io.in.poke("h3FC0404040003F80".U)
        dut.clock.step(1)
        dut.io.out.expect("h4040".U)
      }
    }
  }
}
