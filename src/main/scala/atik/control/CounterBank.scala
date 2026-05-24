package atik.control

import atik._
import chisel3._
import chisel3.util._

class CounterBank(params: AtikParams) extends Module {
  val io = IO(new Bundle {
    val clear = Input(Bool())
    val events = Input(new AtikCounterEvent(params))
    val readIndex = Input(UInt(params.counterIdxBits.W))
    val readData = Output(UInt(params.xLen.W))
  })

  private val counters = RegInit(VecInit(Seq.fill(params.counterCount)(0.U(params.xLen.W))))

  when(io.clear) {
    for (i <- 0 until params.counterCount) {
      counters(i) := 0.U
    }
  }.otherwise {
    when(io.events.totalActive) { counters(AtikCounters.totalCycles) := counters(AtikCounters.totalCycles) + 1.U }
    when(io.events.computeActive) { counters(AtikCounters.computeCycles) := counters(AtikCounters.computeCycles) + 1.U }
    when(io.events.dmaReadActive) { counters(AtikCounters.dmaReadCycles) := counters(AtikCounters.dmaReadCycles) + 1.U }
    when(io.events.dmaWriteActive) { counters(AtikCounters.dmaWriteCycles) := counters(AtikCounters.dmaWriteCycles) + 1.U }
    when(io.events.meshActive) { counters(AtikCounters.meshActiveCycles) := counters(AtikCounters.meshActiveCycles) + 1.U }
    when(io.events.meshIdle) { counters(AtikCounters.meshIdleCycles) := counters(AtikCounters.meshIdleCycles) + 1.U }
    when(io.events.softmaxActive) { counters(AtikCounters.softmaxCycles) := counters(AtikCounters.softmaxCycles) + 1.U }
    when(io.events.dmaStall) { counters(AtikCounters.dmaStallCycles) := counters(AtikCounters.dmaStallCycles) + 1.U }
    when(io.events.sramStall) { counters(AtikCounters.sramStallCycles) := counters(AtikCounters.sramStallCycles) + 1.U }
    counters(AtikCounters.bytesRead) := counters(AtikCounters.bytesRead) + io.events.bytesRead
    counters(AtikCounters.bytesWritten) := counters(AtikCounters.bytesWritten) + io.events.bytesWritten
    counters(AtikCounters.tilesLoaded) := counters(AtikCounters.tilesLoaded) + io.events.tilesLoaded
    counters(AtikCounters.tilesComputed) := counters(AtikCounters.tilesComputed) + io.events.tilesComputed
  }

  io.readData := Mux(
    io.readIndex < params.counterCount.U,
    counters(io.readIndex),
    "hBAD0BAD0BAD0BAD0".U(params.xLen.W)
  )
}
