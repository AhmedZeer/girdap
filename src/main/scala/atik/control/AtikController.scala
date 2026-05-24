package atik.control

import atik._
import chisel3._
import chisel3.util._

class AtikController(params: AtikParams) extends Module {
  val io = IO(new Bundle {
    val setDesc = Input(Bool())
    val descAddr = Input(UInt(params.addrBits.W))
    val run = Input(Bool())
    val resetCore = Input(Bool())

    val descriptorStart = Output(Bool())
    val descriptorAddr = Output(UInt(params.addrBits.W))
    val descriptorDone = Input(Bool())
    val descriptorError = Input(Bool())
    val descriptor = Input(new AtikDescriptor(params))

    val matmulStart = Output(Bool())
    val attentionStart = Output(Bool())
    val activeDesc = Output(new AtikDescriptor(params))
    val causal = Output(Bool())

    val matmulDone = Input(Bool())
    val matmulError = Input(UInt(8.W))
    val matmulEvents = Input(new AtikCounterEvent(params))
    val attentionDone = Input(Bool())
    val attentionError = Input(UInt(8.W))
    val attentionEvents = Input(new AtikCounterEvent(params))

    val status = Output(new AtikStatusState)
    val events = Output(new AtikCounterEvent(params))
  })

  private val (sIdle :: sStartRead :: sReadDesc :: sLaunchMatmul :: sRunMatmul ::
    sLaunchAttention :: sRunAttention :: sDone :: sError :: Nil) = Enum(9)
  private val state = RegInit(sIdle)
  private val descAddrReg = RegInit(0.U(params.addrBits.W))
  private val descReg = RegInit(0.U.asTypeOf(new AtikDescriptor(params)))
  private val doneReg = RegInit(false.B)
  private val errorReg = RegInit(AtikStatus.ok)
  private val causalReg = RegInit(false.B)

  private val busy = state =/= sIdle && state =/= sDone && state =/= sError
  private val isMatmul = io.descriptor.op === AtikOps.matmul
  private val isAttention = io.descriptor.op === AtikOps.attention || io.descriptor.op === AtikOps.causalAttention
  private val attentionSupported = if (params.enableAttention) isAttention else false.B
  private val badOp = !isMatmul && !attentionSupported

  io.descriptorStart := false.B
  io.descriptorAddr := descAddrReg
  io.matmulStart := false.B
  io.attentionStart := false.B
  io.activeDesc := descReg
  io.causal := causalReg

  private val event = WireDefault(0.U.asTypeOf(new AtikCounterEvent(params)))
  when(state === sRunMatmul) {
    event := io.matmulEvents
  }.elsewhen(state === sRunAttention) {
    event := io.attentionEvents
  }
  event.totalActive := busy
  io.events := event

  io.status.busy := busy
  io.status.done := doneReg
  io.status.error := errorReg

  when(io.resetCore) {
    state := sIdle
    descAddrReg := 0.U
    descReg := 0.U.asTypeOf(new AtikDescriptor(params))
    doneReg := false.B
    errorReg := AtikStatus.ok
    causalReg := false.B
  }.otherwise {
    when(io.setDesc && !busy) {
      descAddrReg := io.descAddr
      doneReg := false.B
      errorReg := AtikStatus.ok
    }

    switch(state) {
      is(sIdle) {
        when(io.run) {
          doneReg := false.B
          when(descAddrReg === 0.U) {
            errorReg := AtikStatus.badAddr
            state := sError
          }.otherwise {
            errorReg := AtikStatus.ok
            state := sStartRead
          }
        }
      }
      is(sDone) {
        when(io.run) {
          doneReg := false.B
          state := sStartRead
        }
      }
      is(sError) {
        when(io.run) {
          doneReg := false.B
          when(descAddrReg === 0.U) {
            errorReg := AtikStatus.badAddr
          }.otherwise {
            errorReg := AtikStatus.ok
            state := sStartRead
          }
        }
      }
      is(sStartRead) {
        io.descriptorStart := true.B
        state := sReadDesc
      }
      is(sReadDesc) {
        when(io.descriptorDone) {
          when(io.descriptorError) {
            errorReg := AtikStatus.dma
            state := sError
          }.elsewhen(badOp) {
            errorReg := AtikStatus.badOp
            state := sError
          }.elsewhen(isMatmul) {
            descReg := io.descriptor
            causalReg := false.B
            state := sLaunchMatmul
          }.elsewhen(attentionSupported) {
            descReg := io.descriptor
            causalReg := io.descriptor.op === AtikOps.causalAttention
            state := sLaunchAttention
          }
        }
      }
      is(sLaunchMatmul) {
        io.matmulStart := true.B
        state := sRunMatmul
      }
      is(sRunMatmul) {
        when(io.matmulError =/= AtikStatus.ok) {
          errorReg := io.matmulError
          state := sError
        }.elsewhen(io.matmulDone) {
          doneReg := true.B
          errorReg := AtikStatus.ok
          state := sDone
        }
      }
      is(sLaunchAttention) {
        io.attentionStart := true.B
        state := sRunAttention
      }
      is(sRunAttention) {
        when(io.attentionError =/= AtikStatus.ok) {
          errorReg := io.attentionError
          state := sError
        }.elsewhen(io.attentionDone) {
          doneReg := true.B
          errorReg := AtikStatus.ok
          state := sDone
        }
      }
    }
  }
}
