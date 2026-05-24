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
  private val attention = Module(new AttentionController(params))
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
  controller.io.attentionDone := attention.io.done
  controller.io.attentionError := attention.io.error
  controller.io.attentionEvents := attention.io.events

  descriptorReader.io.start := controller.io.descriptorStart
  descriptorReader.io.addr := controller.io.descriptorAddr
  descriptorReader.io.readCmd <> descriptorDma.io.cmd
  descriptorReader.io.readBeat <> descriptorDma.io.out

  private val readOwnerDesc = 0.U(2.W)
  private val readOwnerMatmul = 1.U(2.W)
  private val readOwnerAttention = 2.U(2.W)
  private val readOwner = RegInit(readOwnerDesc)
  private val descReadSelected = descriptorDma.io.memReq.valid
  private val matmulReadSelected = !descReadSelected && matmul.io.memReadReq.valid
  private val attentionReadSelected = !descReadSelected && !matmulReadSelected && attention.io.memReadReq.valid

  io.memReadReq.valid := descReadSelected || matmulReadSelected || attentionReadSelected
  io.memReadReq.bits := MuxCase(attention.io.memReadReq.bits, Seq(
    descReadSelected -> descriptorDma.io.memReq.bits,
    matmulReadSelected -> matmul.io.memReadReq.bits
  ))
  descriptorDma.io.memReq.ready := descReadSelected && io.memReadReq.ready
  matmul.io.memReadReq.ready := matmulReadSelected && io.memReadReq.ready
  attention.io.memReadReq.ready := attentionReadSelected && io.memReadReq.ready

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
  attention.io.memReadResp.valid := io.memReadResp.valid && readOwner === readOwnerAttention
  attention.io.memReadResp.bits := io.memReadResp.bits
  io.memReadResp.ready := MuxCase(attention.io.memReadResp.ready, Seq(
    (readOwner === readOwnerDesc) -> descriptorDma.io.memResp.ready,
    (readOwner === readOwnerMatmul) -> matmul.io.memReadResp.ready
  ))

  private val matmulWriteSelected = matmul.io.memWriteReq.valid
  private val attentionWriteSelected = !matmulWriteSelected && attention.io.memWriteReq.valid
  private val writeOwnerMatmul = 0.U(1.W)
  private val writeOwnerAttention = 1.U(1.W)
  private val writeOwner = RegInit(writeOwnerMatmul)

  io.memWriteReq.valid := matmulWriteSelected || attentionWriteSelected
  io.memWriteReq.bits := Mux(matmulWriteSelected, matmul.io.memWriteReq.bits, attention.io.memWriteReq.bits)
  matmul.io.memWriteReq.ready := matmulWriteSelected && io.memWriteReq.ready
  attention.io.memWriteReq.ready := attentionWriteSelected && io.memWriteReq.ready
  io.memWriteData := Mux(matmulWriteSelected, matmul.io.memWriteData, attention.io.memWriteData)
  io.memWriteMask := Mux(matmulWriteSelected, matmul.io.memWriteMask, attention.io.memWriteMask)

  when(io.memWriteReq.fire) {
    writeOwner := Mux(matmulWriteSelected, writeOwnerMatmul, writeOwnerAttention)
  }

  matmul.io.memWriteResp.valid := io.memWriteResp.valid && writeOwner === writeOwnerMatmul
  matmul.io.memWriteResp.bits := io.memWriteResp.bits
  attention.io.memWriteResp.valid := io.memWriteResp.valid && writeOwner === writeOwnerAttention
  attention.io.memWriteResp.bits := io.memWriteResp.bits
  io.memWriteResp.ready := Mux(writeOwner === writeOwnerMatmul, matmul.io.memWriteResp.ready, attention.io.memWriteResp.ready)

  matmul.io.start := controller.io.matmulStart
  matmul.io.desc := controller.io.activeDesc

  private val matmulUsesMesh = matmul.io.meshActive
  sharedMesh.io.a := Mux(matmulUsesMesh, matmul.io.meshA, attention.io.meshA)
  sharedMesh.io.b := Mux(matmulUsesMesh, matmul.io.meshB, attention.io.meshB)
  sharedMesh.io.accIn := Mux(matmulUsesMesh, matmul.io.meshAccIn, attention.io.meshAccIn)
  matmul.io.meshOut := sharedMesh.io.out
  attention.io.meshOut := sharedMesh.io.out

  attention.io.start := controller.io.attentionStart
  attention.io.desc := controller.io.activeDesc
  attention.io.causal := controller.io.causal

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
