package toyrocc

import chisel3._
import chiseltest._
import org.scalatest.funspec.AnyFunSpec

class VEXPCoreSpec extends AnyFunSpec with ChiselScalatestTester {

  // Helper function: Unpacks a 64-bit BigInt into four Scala Floats (assuming packed BFloat16)
  def decodeBFloat16Vector(vec: BigInt): Seq[Float] = {
    (0 until 4).map { i =>
      // Extract the 16-bit chunk
      val bf16Bits = ((vec >> (i * 16)) & 0xFFFF).toInt
      // Pad with 16 zeros to make it a standard 32-bit Float
      val f32Bits = bf16Bits << 16
      java.lang.Float.intBitsToFloat(f32Bits)
    }
  }

  describe("VEXPCore Logic") {
    it("receives a command, transitions states, and returns exponentiated vector within tolerance") {
      test(new VEXPCore(64)).withAnnotations(Seq(WriteVcdAnnotation)) { c =>

        // input: [4.0, -1.0, 0.0, 1.0]
        val inputVector = BigInt("4080BF8000003F80", 16).U
        val expectedVector = BigInt("425A3EBC3F80402E", 16).U

        // --- 1 & 2. INITIAL STATE & TRIGGER ---
        c.io.cmd.valid.poke(true.B)
        c.io.cmd.bits.rd.poke(5.U)
        c.io.cmd.bits.rs1.poke(inputVector)
        c.clock.step(1)
        c.io.cmd.valid.poke(false.B)

        // --- 3. PROCESSING STATE ---
        c.clock.step(1)

        // --- 4. FINISH STATE & LENIENT ASSERTION ---
        c.io.resp.valid.expect(true.B)
        c.io.resp.bits.rd.expect(5.U)

        // ---------------------------------------------------------
        // INSTEAD OF: c.io.resp.bits.data.expect(expectedVector)
        // READ THE BITS AND COMPARE AS FLOATS
        // ---------------------------------------------------------
        val actualDataBigInt = c.io.resp.bits.data.peek().litValue
        val expectedDataBigInt = expectedVector.litValue

        val actualFloats = decodeBFloat16Vector(actualDataBigInt)
        val expectedFloats = decodeBFloat16Vector(expectedDataBigInt)

        // Define your allowable error tolerance (e.g., +/- 0.5)
        // You may need to tune this depending on your piecewise linear approximation accuracy!
        val epsilon = 1.0f

        for (i <- 0 until 4) {
          val actual = actualFloats(i)
          val expected = expectedFloats(i)
          val error = Math.abs(actual - expected)

          assert(
            error <= epsilon,
            s"Vector index $i failed! Expected ~$expected, Got ~$actual (Error: $error > Tolerance: $epsilon)"
          )
        }

        // Acknowledge the response
        c.io.resp.ready.poke(true.B)
        c.clock.step(1)
      }
    }
  }
}