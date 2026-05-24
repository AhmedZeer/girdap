package atik

import chisel3._

object AtikOpcodes {
  val setDesc = 0.U(7.W)
  val run = 1.U(7.W)
  val status = 2.U(7.W)
  val readCounter = 3.U(7.W)
  val clearCounters = 4.U(7.W)
  val reset = 5.U(7.W)
}

object AtikOps {
  val matmul = 0.U(32.W)
  val attention = 1.U(32.W)
  val causalAttention = 2.U(32.W)
}

object AtikStatus {
  val ok = 0.U(8.W)
  val badOp = 1.U(8.W)
  val badDims = 2.U(8.W)
  val badAddr = 3.U(8.W)
  val unsupportedConfig = 4.U(8.W)
  val dma = 5.U(8.W)
  val internal = 6.U(8.W)
  val timeout = 7.U(8.W)
}

object AtikCounters {
  val totalCycles = 0
  val computeCycles = 1
  val dmaReadCycles = 2
  val dmaWriteCycles = 3
  val meshActiveCycles = 4
  val meshIdleCycles = 5
  val bytesRead = 6
  val bytesWritten = 7
  val softmaxCycles = 8
  val tilesLoaded = 9
  val tilesComputed = 10
  val dmaStallCycles = 11
  val sramStallCycles = 12
}
