package atik.rocc

import atik._
import atik.top._
import chisel3._
import chisel3.util._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import org.chipsalliance.cde.config.{Config, Parameters}

class AtikRoCC(
  opcodes: OpcodeSet,
  val atikParams: AtikParams = AtikParams(),
  val clientName: String = "AtikRoCC",
  val numTLSourceIds: Int = 2
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new AtikRoCCModule(this)
  override val atlNode = TLClientNode(
    Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1(
      name = clientName,
      sourceId = IdRange(0, numTLSourceIds)
    ))))
  )
}

class AtikRoCCModule(outer: AtikRoCC)(implicit p: Parameters)
    extends LazyRoCCModuleImp(outer)
    with HasCoreParameters
    with HasL1CacheParameters {

  val cacheParams = tileParams.dcache.get

  private val params = outer.atikParams.copy(xLen = xLen, memDataBits = cacheDataBits)
  private val readLgSize = log2Ceil(params.beatBytes).U
  private val tlSourceIdxWidth = math.max(1, log2Ceil(outer.numTLSourceIds))
  private val readTlSourceId = 0
  private val writeTlSourceId = 1

  private val top = Module(new AtikTop(params))
  private val router = Module(new AtikCommandRouter(params))

  router.io.cmd.valid := io.cmd.valid
  router.io.cmd.bits.funct := io.cmd.bits.inst.funct
  router.io.cmd.bits.rd := io.cmd.bits.inst.rd
  router.io.cmd.bits.rs1 := io.cmd.bits.rs1
  router.io.cmd.bits.rs2 := io.cmd.bits.rs2
  io.cmd.ready := router.io.cmd.ready

  io.resp.valid := router.io.resp.valid
  router.io.resp.ready := io.resp.ready
  io.resp.bits.rd := router.io.resp.bits.rd
  io.resp.bits.data := router.io.resp.bits.data

  top.io.setDesc := router.io.setDesc
  top.io.descAddr := router.io.descAddr
  top.io.run := router.io.run
  top.io.resetCore := router.io.resetCore
  top.io.clearCounters := router.io.clearCounters
  top.io.counterIndex := router.io.counterIndex
  router.io.statusWord := top.io.statusWord
  router.io.counterValue := top.io.counterValue

  io.busy := top.io.busy || router.io.busy
  io.interrupt := false.B

  val (tlOut, edgesOut) = outer.atlNode.out(0)
  private val dHasData = edgesOut.hasData(tlOut.d.bits)
  private val dSource = tlOut.d.bits.source(tlSourceIdxWidth - 1, 0)
  private val dIsReadResp = dHasData && tlOut.d.bits.opcode === TLMessages.AccessAckData
  private val dIsWriteAck = !dHasData && tlOut.d.bits.opcode === TLMessages.AccessAck
  private val dError = tlOut.d.bits.denied

  private val memOutstanding = RegInit(false.B)
  private val outstandingRead = RegInit(false.B)
  private val outstandingWrite = RegInit(false.B)
  private val readSelected = top.io.memReadReq.valid
  private val writeSelected = !readSelected && top.io.memWriteReq.valid

  private val getBits = edgesOut.Get(
    fromSource = readTlSourceId.U(tlSourceIdxWidth.W),
    toAddress = top.io.memReadReq.bits.addr,
    lgSize = readLgSize
  )._2
  private val putBits = edgesOut.Put(
    fromSource = writeTlSourceId.U(tlSourceIdxWidth.W),
    toAddress = top.io.memWriteReq.bits.addr,
    lgSize = readLgSize,
    data = top.io.memWriteData,
    mask = top.io.memWriteMask
  )._2

  tlOut.a.valid := !memOutstanding && (readSelected || writeSelected)
  tlOut.a.bits := Mux(writeSelected, putBits, getBits)
  tlOut.d.ready := memOutstanding && Mux(
    outstandingRead,
    top.io.memReadResp.ready,
    Mux(outstandingWrite, top.io.memWriteResp.ready, false.B)
  )

  top.io.memReadReq.ready := !memOutstanding && readSelected && tlOut.a.ready
  top.io.memWriteReq.ready := !memOutstanding && writeSelected && tlOut.a.ready

  tlOut.b.ready := true.B
  tlOut.c.valid := false.B
  tlOut.c.bits := DontCare
  tlOut.e.valid := false.B
  tlOut.e.bits := DontCare

  when(tlOut.a.fire) {
    memOutstanding := true.B
    outstandingRead := readSelected
    outstandingWrite := writeSelected
  }

  top.io.memReadResp.valid := tlOut.d.valid && memOutstanding && outstandingRead
  top.io.memReadResp.bits.data := tlOut.d.bits.data(params.memDataBits - 1, 0)
  top.io.memReadResp.bits.error := dError || dSource =/= readTlSourceId.U || !dIsReadResp
  top.io.memWriteResp.valid := tlOut.d.valid && memOutstanding && outstandingWrite
  top.io.memWriteResp.bits.error := dError || dSource =/= writeTlSourceId.U || !dIsWriteAck

  when((top.io.memReadResp.fire && outstandingRead) || (top.io.memWriteResp.fire && outstandingWrite)) {
    memOutstanding := false.B
    outstandingRead := false.B
    outstandingWrite := false.B
  }
}

class WithAtikRoCC(params: AtikParams = AtikParams()) extends Config((site, here, up) => {
  case BuildRoCC => Seq(
    (p: Parameters) => LazyModule(new AtikRoCC(OpcodeSet.custom0, params)(p))
  )
})

class WithAtik2x2RoCC extends WithAtikRoCC(AtikParams(meshRows = 2, meshCols = 2))
class WithAtik4x4RoCC extends WithAtikRoCC(AtikParams(meshRows = 4, meshCols = 4))
class WithAtik8x8RoCC extends WithAtikRoCC(AtikParams(meshRows = 8, meshCols = 8))
