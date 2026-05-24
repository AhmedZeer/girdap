package atik.top

import atik._
import atik.control._
import atik.compute._
import atik.memory._
import chisel3._
import chisel3.util._

class AtikCoreIO(params: AtikParams) extends Bundle {
  val setDesc = Input(Bool())
  val descAddr = Input(UInt(params.addrBits.W))
  val run = Input(Bool())
  val resetCore = Input(Bool())
  val clearCounters = Input(Bool())
  val counterIndex = Input(UInt(params.counterIdxBits.W))

  val statusWord = Output(UInt(params.xLen.W))
  val counterValue = Output(UInt(params.xLen.W))
  val busy = Output(Bool())

  val memReadReq = Decoupled(new DmaBeatRequest(params))
  val memReadResp = Flipped(Decoupled(new DmaBeatResponse(params)))
  val memWriteReq = Decoupled(new DmaBeatRequest(params))
  val memWriteData = Output(UInt(params.memDataBits.W))
  val memWriteMask = Output(UInt((params.memDataBits / 8).W))
  val memWriteResp = Flipped(Decoupled(new DmaWriteResponse))
}

class AtikCore(params: AtikParams) extends Module {
  val io = IO(new AtikCoreIO(params))

  private val descriptorReader = Module(new DescriptorReader(params))
  private val descriptorDma = Module(new DmaReader(params))
  private val controller = Module(new AtikController(params))
  private val matmul = Module(new MatmulController(params))
  private val attentionOpt = if (params.enableAttention) Some(Module(new AttentionController(params))) else None
  private val sharedMesh = Module(new MacMesh(params))
  private val counters = Module(new CounterBank(params))
  private val statusRegs = Module(new StatusRegs(params))

  controller.io.setDesc := io.setDesc
  controller.io.descAddr := io.descAddr
  controller.io.run := io.run
  controller.io.resetCore := io.resetCore
  controller.io.descriptorDone := descriptorReader.io.done
  controller.io.descriptorError := descriptorReader.io.error
  controller.io.descriptor := descriptorReader.io.desc
  controller.io.matmulDone := matmul.io.done
  controller.io.matmulError := matmul.io.error
  controller.io.matmulEvents := matmul.io.events
  controller.io.attentionDone := false.B
  controller.io.attentionError := AtikStatus.unsupportedConfig
  controller.io.attentionEvents := 0.U.asTypeOf(new AtikCounterEvent(params))
  attentionOpt.foreach { attention =>
    controller.io.attentionDone := attention.io.done
    controller.io.attentionError := attention.io.error
    controller.io.attentionEvents := attention.io.events
  }

  descriptorReader.io.start := controller.io.descriptorStart
  descriptorReader.io.addr := controller.io.descriptorAddr
  descriptorReader.io.readCmd <> descriptorDma.io.cmd
  descriptorReader.io.readBeat <> descriptorDma.io.out

  private val readOwnerDesc = 0.U(2.W)
  private val readOwnerMatmul = 1.U(2.W)
  private val readOwnerAttention = 2.U(2.W)
  private val readOwner = RegInit(readOwnerDesc)
  private val attentionReadValid = WireDefault(false.B)
  private val attentionReadBits = WireDefault(0.U.asTypeOf(new DmaBeatRequest(params)))
  attentionOpt.foreach { attention =>
    attentionReadValid := attention.io.memReadReq.valid
    attentionReadBits := attention.io.memReadReq.bits
  }

  private val descReadSelected = descriptorDma.io.memReq.valid
  private val matmulReadSelected = !descReadSelected && matmul.io.memReadReq.valid
  private val attentionReadSelected = !descReadSelected && !matmulReadSelected && attentionReadValid

  io.memReadReq.valid := descReadSelected || matmulReadSelected || attentionReadSelected
  io.memReadReq.bits := MuxCase(attentionReadBits, Seq(
    descReadSelected -> descriptorDma.io.memReq.bits,
    matmulReadSelected -> matmul.io.memReadReq.bits
  ))
  descriptorDma.io.memReq.ready := descReadSelected && io.memReadReq.ready
  matmul.io.memReadReq.ready := matmulReadSelected && io.memReadReq.ready
  attentionOpt.foreach { attention =>
    attention.io.memReadReq.ready := attentionReadSelected && io.memReadReq.ready
  }

  when(io.memReadReq.fire) {
    readOwner := MuxCase(readOwnerAttention, Seq(
      descReadSelected -> readOwnerDesc,
      matmulReadSelected -> readOwnerMatmul
    ))
  }

  descriptorDma.io.memResp.valid := io.memReadResp.valid && readOwner === readOwnerDesc
  descriptorDma.io.memResp.bits := io.memReadResp.bits
  matmul.io.memReadResp.valid := io.memReadResp.valid && readOwner === readOwnerMatmul
  matmul.io.memReadResp.bits := io.memReadResp.bits
  private val attentionReadRespReady = WireDefault(false.B)
  attentionOpt.foreach { attention =>
    attention.io.memReadResp.valid := io.memReadResp.valid && readOwner === readOwnerAttention
    attention.io.memReadResp.bits := io.memReadResp.bits
    attentionReadRespReady := attention.io.memReadResp.ready
  }
  io.memReadResp.ready := MuxCase(attentionReadRespReady, Seq(
    (readOwner === readOwnerDesc) -> descriptorDma.io.memResp.ready,
    (readOwner === readOwnerMatmul) -> matmul.io.memReadResp.ready
  ))

  private val attentionWriteValid = WireDefault(false.B)
  private val attentionWriteBits = WireDefault(0.U.asTypeOf(new DmaBeatRequest(params)))
  private val attentionWriteData = WireDefault(0.U(params.memDataBits.W))
  private val attentionWriteMask = WireDefault(0.U((params.memDataBits / 8).W))
  attentionOpt.foreach { attention =>
    attentionWriteValid := attention.io.memWriteReq.valid
    attentionWriteBits := attention.io.memWriteReq.bits
    attentionWriteData := attention.io.memWriteData
    attentionWriteMask := attention.io.memWriteMask
  }

  private val matmulWriteSelected = matmul.io.memWriteReq.valid
  private val attentionWriteSelected = !matmulWriteSelected && attentionWriteValid
  private val writeOwnerMatmul = 0.U(1.W)
  private val writeOwnerAttention = 1.U(1.W)
  private val writeOwner = RegInit(writeOwnerMatmul)

  io.memWriteReq.valid := matmulWriteSelected || attentionWriteSelected
  io.memWriteReq.bits := Mux(matmulWriteSelected, matmul.io.memWriteReq.bits, attentionWriteBits)
  matmul.io.memWriteReq.ready := matmulWriteSelected && io.memWriteReq.ready
  attentionOpt.foreach { attention =>
    attention.io.memWriteReq.ready := attentionWriteSelected && io.memWriteReq.ready
  }
  io.memWriteData := Mux(matmulWriteSelected, matmul.io.memWriteData, attentionWriteData)
  io.memWriteMask := Mux(matmulWriteSelected, matmul.io.memWriteMask, attentionWriteMask)

  when(io.memWriteReq.fire) {
    writeOwner := Mux(matmulWriteSelected, writeOwnerMatmul, writeOwnerAttention)
  }

  matmul.io.memWriteResp.valid := io.memWriteResp.valid && writeOwner === writeOwnerMatmul
  matmul.io.memWriteResp.bits := io.memWriteResp.bits
  private val attentionWriteRespReady = WireDefault(false.B)
  attentionOpt.foreach { attention =>
    attention.io.memWriteResp.valid := io.memWriteResp.valid && writeOwner === writeOwnerAttention
    attention.io.memWriteResp.bits := io.memWriteResp.bits
    attentionWriteRespReady := attention.io.memWriteResp.ready
  }
  io.memWriteResp.ready := Mux(writeOwner === writeOwnerMatmul, matmul.io.memWriteResp.ready, attentionWriteRespReady)

  matmul.io.start := controller.io.matmulStart
  matmul.io.desc := controller.io.activeDesc

  private val attentionMeshA = WireDefault(0.U.asTypeOf(Vec(params.meshRows, SInt(params.fixedBits.W))))
  private val attentionMeshB = WireDefault(0.U.asTypeOf(Vec(params.meshCols, SInt(params.fixedBits.W))))
  private val attentionMeshAccIn = WireDefault(0.U.asTypeOf(Vec(params.meshRows, Vec(params.meshCols, SInt(params.accumBits.W)))))
  attentionOpt.foreach { attention =>
    attentionMeshA := attention.io.meshA
    attentionMeshB := attention.io.meshB
    attentionMeshAccIn := attention.io.meshAccIn
    attention.io.meshOut := sharedMesh.io.out
    attention.io.start := controller.io.attentionStart
    attention.io.desc := controller.io.activeDesc
    attention.io.causal := controller.io.causal
  }

  private val matmulUsesMesh = if (params.enableAttention) matmul.io.meshActive else true.B
  sharedMesh.io.a := Mux(matmulUsesMesh, matmul.io.meshA, attentionMeshA)
  sharedMesh.io.b := Mux(matmulUsesMesh, matmul.io.meshB, attentionMeshB)
  sharedMesh.io.accIn := Mux(matmulUsesMesh, matmul.io.meshAccIn, attentionMeshAccIn)
  matmul.io.meshOut := sharedMesh.io.out

  private val counterEvents = WireDefault(controller.io.events)
  counterEvents.dmaReadActive := controller.io.events.dmaReadActive || descriptorDma.io.active
  counterEvents.bytesRead := controller.io.events.bytesRead + Mux(descriptorDma.io.beatIssued, params.beatBytes.U(params.xLen.W), 0.U)

  counters.io.clear := io.clearCounters || io.resetCore
  counters.io.events := counterEvents
  counters.io.readIndex := io.counterIndex
  io.counterValue := counters.io.readData

  statusRegs.io.state := controller.io.status
  io.statusWord := statusRegs.io.word
  io.busy := controller.io.status.busy
}
