package girdap

import chisel3._
import chisel3.util.Cat

/** Flattened top module for taking the 2x2 output-stationary systolic core
  * through standalone ASIC flows such as OpenLane.
  */
class SystolicArrayCore2x2OpenLaneTop extends Module {
  private val precision = 16
  private val nRows = 2
  private val nCols = 2
  private val maxK = 64
  private val kWidth = 7
  private val accumPrec = 2 * precision

  val io = IO(new Bundle {
    val cmdValid = Input(Bool())
    val cmdReady = Output(Bool())
    val cmdClear = Input(Bool())
    val k = Input(UInt(kWidth.W))

    val inValid = Input(Bool())
    val inReady = Output(Bool())
    val inLeft = Input(UInt((nRows * precision).W))
    val inTop = Input(UInt((nCols * precision).W))

    val out = Output(UInt((nRows * nCols * accumPrec).W))
    val busy = Output(Bool())
    val done = Output(Bool())
  })

  val core = Module(new SystolicArrayCore(
    precision = precision,
    nRows = nRows,
    nCols = nCols,
    maxK = maxK
  ))

  core.io.cmdValid := io.cmdValid
  io.cmdReady := core.io.cmdReady
  core.io.cmdClear := io.cmdClear
  core.io.k := io.k

  core.io.inValid := io.inValid
  io.inReady := core.io.inReady
  for (i <- 0 until nRows) {
    core.io.in_left(i) := io.inLeft((i + 1) * precision - 1, i * precision)
  }
  for (j <- 0 until nCols) {
    core.io.in_top(j) := io.inTop((j + 1) * precision - 1, j * precision)
  }

  io.out := Cat(
    core.io.out(1)(1),
    core.io.out(1)(0),
    core.io.out(0)(1),
    core.io.out(0)(0)
  )
  io.busy := core.io.busy
  io.done := core.io.done
}

object EmitSystolicArrayCore2x2OpenLane extends App {
  val targetDir =
    if (args.nonEmpty) args(0)
    else "/home/ubuntu/firesim_build_disk/openlane_girdap/src"

  circt.stage.ChiselStage.emitSystemVerilogFile(
    new SystolicArrayCore2x2OpenLaneTop,
    Array("--target-dir", targetDir),
    Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    )
  )
}
