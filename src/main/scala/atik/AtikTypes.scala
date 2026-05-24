package atik

import chisel3._
import chisel3.util.Cat

class AtikDescriptor(params: AtikParams) extends Bundle {
  val op = UInt(32.W)
  val reserved0 = UInt(32.W)

  val aAddr = UInt(params.addrBits.W)
  val bAddr = UInt(params.addrBits.W)
  val vAddr = UInt(params.addrBits.W)
  val outAddr = UInt(params.addrBits.W)

  val m = UInt(32.W)
  val n = UInt(32.W)
  val k = UInt(32.W)
  val kvLen = UInt(32.W)

  val lda = UInt(32.W)
  val ldb = UInt(32.W)
  val ldv = UInt(32.W)
  val ldout = UInt(32.W)

  val scaleBf16 = UInt(16.W)
  val reserved1 = UInt(16.W)
  val reserved2 = UInt(32.W)
}

class AtikHostCommand(params: AtikParams) extends Bundle {
  val funct = UInt(7.W)
  val rd = UInt(5.W)
  val rs1 = UInt(params.xLen.W)
  val rs2 = UInt(params.xLen.W)
}

class AtikHostResponse(params: AtikParams) extends Bundle {
  val rd = UInt(5.W)
  val data = UInt(params.xLen.W)
}

class AtikStatusState extends Bundle {
  val busy = Bool()
  val done = Bool()
  val error = UInt(8.W)
}

object AtikStatusWord {
  def pack(status: AtikStatusState, xLen: Int): UInt = {
    Cat(0.U((xLen - 16).W), status.error, 0.U(5.W), status.error =/= AtikStatus.ok, status.done, status.busy)
  }
}

class AtikCounterEvent(params: AtikParams) extends Bundle {
  val totalActive = Bool()
  val computeActive = Bool()
  val dmaReadActive = Bool()
  val dmaWriteActive = Bool()
  val meshActive = Bool()
  val meshIdle = Bool()
  val softmaxActive = Bool()
  val dmaStall = Bool()
  val sramStall = Bool()
  val bytesRead = UInt(params.xLen.W)
  val bytesWritten = UInt(params.xLen.W)
  val tilesLoaded = UInt(params.xLen.W)
  val tilesComputed = UInt(params.xLen.W)
}

class DmaReadCommand(params: AtikParams) extends Bundle {
  val base = UInt(params.addrBits.W)
  val beats = UInt(params.dmaBeatsBits.W)
}

class DmaWriteCommand(params: AtikParams) extends Bundle {
  val base = UInt(params.addrBits.W)
  val beats = UInt(params.dmaBeatsBits.W)
}

class DmaBeatRequest(params: AtikParams) extends Bundle {
  val addr = UInt(params.addrBits.W)
}

class DmaBeatResponse(params: AtikParams) extends Bundle {
  val data = UInt(params.memDataBits.W)
  val error = Bool()
}

class DmaReadBeat(params: AtikParams) extends Bundle {
  val data = UInt(params.memDataBits.W)
  val index = UInt(params.dmaBeatsBits.W)
  val last = Bool()
  val error = Bool()
}

class DmaWriteBeat(params: AtikParams) extends Bundle {
  val data = UInt(params.memDataBits.W)
  val mask = UInt((params.memDataBits / 8).W)
  val last = Bool()
}

class DmaWriteResponse extends Bundle {
  val error = Bool()
}

object AtikTypeDefaults {
  def descriptor(params: AtikParams): AtikDescriptor = {
    val desc = Wire(new AtikDescriptor(params))
    desc := 0.U.asTypeOf(new AtikDescriptor(params))
    desc
  }

  def counterEvent(params: AtikParams): AtikCounterEvent = {
    val event = Wire(new AtikCounterEvent(params))
    event := 0.U.asTypeOf(new AtikCounterEvent(params))
    event
  }
}
