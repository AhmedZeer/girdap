package toyrocc

import chisel3._
import chiseltest._
import org.scalatest.funspec.AnyFunSpec

class VEXPCoreSpec extends AnyFunSpec with ChiselScalatestTester {

  describe("VEXPCore Logic") {
    it("receives a command with BFloat16 vectors, transitions states, and returns exponentiated vector") {

      // We test the pure core, passing xLen = 64 manually!
      test(new VEXPCore(64)).withAnnotations(Seq(WriteVcdAnnotation)) { c =>

        // input: [4.0, -1.0, 0.0, 1.0]
        // val inputVector = BigInt("3F003F8000004080", 16).U
         val inputVector = BigInt("4080BF8000003F80", 16).U
        // expected: [~54.5, ~0.367, 1.0, ~2.718]
        // val expectedVector = BigInt("0000402E3F80425A", 16).U
         val expectedVector = BigInt("425A3EBC3F80402E", 16).U

        // --- 1. INITIAL STATE (s_idle) ---
        c.io.cmd.valid.poke(false.B)
        c.io.resp.ready.poke(false.B)
        c.io.cmd.ready.expect(true.B) // Should be ready for a command

        // --- 2. TRIGGER COMMAND ---
        c.io.cmd.valid.poke(true.B)
        c.io.cmd.bits.rd.poke(5.U) // Pass destination register 5
        c.io.cmd.bits.rs1.poke(inputVector) // Pass the BFloat16 vector
        c.clock.step(1)
        c.io.cmd.valid.poke(false.B)

        // --- 3. PROCESSING STATE (s_proc) ---
        c.io.cmd.ready.expect(false.B) // Busy, not ready for new commands
        c.clock.step(1)

        // --- 4. FINISH STATE (s_finish) ---
        c.io.resp.valid.expect(true.B)
        c.io.resp.bits.data.expect(expectedVector)
        c.io.resp.bits.rd.expect(5.U)

        // Acknowledge the response
        c.io.resp.ready.poke(true.B)
        c.clock.step(1)

        // --- 5. BACK TO IDLE ---
        c.io.resp.ready.poke(false.B)
        c.io.cmd.ready.expect(true.B)
      }
    }
  }
}