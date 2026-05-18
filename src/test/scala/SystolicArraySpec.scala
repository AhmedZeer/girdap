package girdap

import chisel3._
import chiseltest._
import org.scalatest.funspec.AnyFunSpec

class SystolicArraySpec extends AnyFunSpec with ChiselScalatestTester {

  describe("ProcessingElement") {
    it("accumulates only on valid cycles and clears correctly") {
      test(new ProcessingElement(inPrec = 8, accumPrec = 16)) { dut =>
        dut.io.clear.poke(true.B)
        dut.io.inValid.poke(false.B)
        dut.io.left.poke(0.U)
        dut.io.top.poke(0.U)
        dut.clock.step(1)
        dut.io.result.expect(0.U)
        dut.io.outValid.expect(false.B)

        dut.io.clear.poke(false.B)
        dut.io.inValid.poke(true.B)
        dut.io.left.poke(3.U)
        dut.io.top.poke(4.U)
        dut.clock.step(1)
        dut.io.result.expect(12.U)
        dut.io.outValid.expect(true.B)

        dut.io.left.poke(2.U)
        dut.io.top.poke(5.U)
        dut.clock.step(1)
        dut.io.result.expect(22.U)
        dut.io.outValid.expect(true.B)

        dut.io.inValid.poke(false.B)
        dut.io.left.poke(100.U)
        dut.io.top.poke(100.U)
        dut.clock.step(1)
        dut.io.result.expect(22.U)
        dut.io.outValid.expect(false.B)

        dut.io.clear.poke(true.B)
        dut.clock.step(1)
        dut.io.result.expect(0.U)
        dut.io.outValid.expect(false.B)
      }
    }
  }

  describe("Controller") {
    it("streams skewed left/top edges for a 2x2 mesh") {
      test(new Controller(precision = 16, nRows = 2, nCols = 2, maxK = 8)) { dut =>
        val k = 3

        dut.io.inValid.poke(false.B)
        dut.io.in_left(0).poke(0.U); dut.io.in_left(1).poke(0.U)
        dut.io.in_top(0).poke(0.U); dut.io.in_top(1).poke(0.U)
        dut.io.cmdValid.poke(false.B)
        dut.io.k.poke(0.U)
        dut.io.clearOnStart.poke(true.B)
        dut.clock.step(1)

        dut.io.k.poke(k.U)
        dut.io.clearOnStart.poke(true.B)
        dut.io.cmdValid.poke(true.B)
        dut.clock.step(1)
        dut.io.cmdValid.poke(false.B)

        val leftSlices = Seq((10, 20), (11, 21), (12, 22))
        val topSlices = Seq((30, 40), (31, 41), (32, 42))

        for (idx <- 0 until k) {
          while (!dut.io.inReady.peek().litToBoolean) {
            dut.clock.step(1)
          }
          dut.io.in_left(0).poke(leftSlices(idx)._1.U)
          dut.io.in_left(1).poke(leftSlices(idx)._2.U)
          dut.io.in_top(0).poke(topSlices(idx)._1.U)
          dut.io.in_top(1).poke(topSlices(idx)._2.U)
          dut.io.inValid.poke(true.B)
          dut.clock.step(1)
        }

        dut.io.inValid.poke(false.B)
        dut.io.in_left(0).poke(0.U); dut.io.in_left(1).poke(0.U)
        dut.io.in_top(0).poke(0.U); dut.io.in_top(1).poke(0.U)

        // One-cycle clear pulse between load and stream.
        dut.io.clear.expect(true.B)
        dut.clock.step(1)

        val expectLeft = Seq(
          (10, 0),
          (11, 20),
          (12, 21),
          (0, 22),
          (0, 0)
        )
        val expectTop = Seq(
          (30, 0),
          (31, 40),
          (32, 41),
          (0, 42),
          (0, 0)
        )
        val expectOutValid = Seq(true, true, true, false, false)

        for (t <- expectLeft.indices) {
          dut.io.out_left(0).expect(expectLeft(t)._1.U)
          dut.io.out_left(1).expect(expectLeft(t)._2.U)
          dut.io.out_top(0).expect(expectTop(t)._1.U)
          dut.io.out_top(1).expect(expectTop(t)._2.U)
          dut.io.outValid.expect(expectOutValid(t).B)
          dut.clock.step(1)
        }

        dut.io.done.expect(true.B)
      }
    }
  }

  describe("SystolicArrayCore") {
    it("computes one 2x2 tile correctly") {
      test(new SystolicArrayCore(precision = 16, nRows = 2, nCols = 2, maxK = 8)) { dut =>
        val k = 3

        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        val leftSlices = Seq((1, 4), (2, 5), (3, 6))
        val topSlices = Seq((7, 8), (9, 10), (11, 12))

        dut.io.inValid.poke(false.B)
        dut.io.in_left(0).poke(0.U); dut.io.in_left(1).poke(0.U)
        dut.io.in_top(0).poke(0.U); dut.io.in_top(1).poke(0.U)
        dut.io.cmdValid.poke(false.B)
        dut.io.k.poke(0.U)
        dut.io.cmdClear.poke(true.B)
        dut.clock.step(1)

        while (!dut.io.cmdReady.peek().litToBoolean) {
          dut.clock.step(1)
        }

        dut.io.k.poke(k.U)
        dut.io.cmdClear.poke(true.B)
        dut.io.cmdValid.poke(true.B)
        dut.clock.step(1)
        dut.io.cmdValid.poke(false.B)

        for (idx <- 0 until k) {
          while (!dut.io.inReady.peek().litToBoolean) {
            dut.clock.step(1)
          }
          dut.io.in_left(0).poke(leftSlices(idx)._1.U)
          dut.io.in_left(1).poke(leftSlices(idx)._2.U)
          dut.io.in_top(0).poke(topSlices(idx)._1.U)
          dut.io.in_top(1).poke(topSlices(idx)._2.U)
          dut.io.inValid.poke(true.B)
          dut.clock.step(1)
        }

        dut.io.inValid.poke(false.B)
        dut.io.in_left(0).poke(0.U); dut.io.in_left(1).poke(0.U)
        dut.io.in_top(0).poke(0.U); dut.io.in_top(1).poke(0.U)

        var waited = 0
        while (!dut.io.done.peek().litToBoolean && waited < 64) {
          dut.clock.step(1)
          waited += 1
        }
        assert(waited < 64, "core did not assert done within timeout")

        dut.io.out(0)(0).expect(58.U)
        dut.io.out(0)(1).expect(64.U)
        dut.io.out(1)(0).expect(139.U)
        dut.io.out(1)(1).expect(154.U)
      }
    }

    it("accumulates across chunked runs when cmdClear is deasserted") {
      test(new SystolicArrayCore(precision = 16, nRows = 2, nCols = 2, maxK = 8)) { dut =>
        def launchRun(k: Int, clear: Boolean, left: Seq[(Int, Int)], top: Seq[(Int, Int)]): Unit = {
          while (!dut.io.cmdReady.peek().litToBoolean) {
            dut.clock.step(1)
          }

          dut.io.k.poke(k.U)
          dut.io.cmdClear.poke(clear.B)
          dut.io.cmdValid.poke(true.B)
          dut.clock.step(1)
          dut.io.cmdValid.poke(false.B)

          for (idx <- 0 until k) {
            while (!dut.io.inReady.peek().litToBoolean) {
              dut.clock.step(1)
            }
            dut.io.in_left(0).poke(left(idx)._1.U)
            dut.io.in_left(1).poke(left(idx)._2.U)
            dut.io.in_top(0).poke(top(idx)._1.U)
            dut.io.in_top(1).poke(top(idx)._2.U)
            dut.io.inValid.poke(true.B)
            dut.clock.step(1)
          }

          dut.io.inValid.poke(false.B)
          dut.io.in_left(0).poke(0.U); dut.io.in_left(1).poke(0.U)
          dut.io.in_top(0).poke(0.U); dut.io.in_top(1).poke(0.U)

          var waited = 0
          while (!dut.io.done.peek().litToBoolean && waited < 64) {
            dut.clock.step(1)
            waited += 1
          }
          assert(waited < 64, "core did not assert done within timeout")
        }

        dut.io.inValid.poke(false.B)
        dut.io.in_left(0).poke(0.U); dut.io.in_left(1).poke(0.U)
        dut.io.in_top(0).poke(0.U); dut.io.in_top(1).poke(0.U)
        dut.io.cmdValid.poke(false.B)
        dut.io.k.poke(0.U)
        dut.io.cmdClear.poke(true.B)
        dut.clock.step(1)

        // Chunk 0 (K=2), clears accumulators.
        launchRun(
          k = 2,
          clear = true,
          left = Seq((1, 4), (2, 5)),
          top = Seq((7, 8), (9, 10))
        )

        dut.io.out(0)(0).expect(25.U)
        dut.io.out(0)(1).expect(28.U)
        dut.io.out(1)(0).expect(73.U)
        dut.io.out(1)(1).expect(82.U)

        // Chunk 1 (K=1), no clear -> accumulates.
        launchRun(
          k = 1,
          clear = false,
          left = Seq((3, 6)),
          top = Seq((11, 12))
        )

        dut.io.out(0)(0).expect(58.U)
        dut.io.out(0)(1).expect(64.U)
        dut.io.out(1)(0).expect(139.U)
        dut.io.out(1)(1).expect(154.U)
      }
    }
  }
}
