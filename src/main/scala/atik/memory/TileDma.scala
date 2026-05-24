package atik.memory

import atik._
import chisel3._
import chisel3.util._

class TileDmaReadCommand(params: AtikParams) extends Bundle {
  val base = UInt(params.addrBits.W)
  val rows = UInt(32.W)
  val cols = UInt(32.W)
  val stride = UInt(32.W)
}

class TileDmaReadElement(params: AtikParams) extends Bundle {
  val data = UInt(params.elemBits.W)
  val row = UInt(32.W)
  val col = UInt(32.W)
  val last = Bool()
  val error = Bool()
}

class TileDmaWriteCommand(params: AtikParams) extends Bundle {
  val base = UInt(params.addrBits.W)
  val rows = UInt(32.W)
  val cols = UInt(32.W)
  val stride = UInt(32.W)
}

class TileDmaReader(params: AtikParams) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new TileDmaReadCommand(params)))
    val memReq = Decoupled(new DmaBeatRequest(params))
    val memResp = Flipped(Decoupled(new DmaBeatResponse(params)))
    val out = Decoupled(new TileDmaReadElement(params))
    val active = Output(Bool())
    val bytesRead = Output(UInt(params.xLen.W))
  })

  private val beatOffsetBits = log2Ceil(params.beatBytes)
  private val elemOffsetBits = log2Ceil(params.bytesPerElem)
  private val elemLaneBits = math.max(1, log2Ceil(params.elemsPerBeat))
  private val sIdle :: sReq :: sWait :: sOut :: Nil = Enum(4)
  private val state = RegInit(sIdle)

  private val base = Reg(UInt(params.addrBits.W))
  private val rows = Reg(UInt(32.W))
  private val cols = Reg(UInt(32.W))
  private val stride = Reg(UInt(32.W))
  private val row = RegInit(0.U(32.W))
  private val col = RegInit(0.U(32.W))
  private val readElemAddr = Reg(UInt(params.addrBits.W))
  private val dataReg = Reg(UInt(params.memDataBits.W))
  private val errorReg = RegInit(false.B)

  private def elemOffset(index: UInt): UInt = (index << elemOffsetBits).pad(params.addrBits)
  private def elemAddr(r: UInt, c: UInt): UInt = base + elemOffset(r * stride + c)
  private def alignedBeat(addr: UInt): UInt = addr & (~((params.beatBytes - 1).U(params.addrBits.W))).asUInt
  private def elemLane(addr: UInt): UInt = addr(beatOffsetBits - 1, elemOffsetBits)
  private def elemFromBeat(data: UInt, lane: UInt): UInt = {
    val elems = Wire(Vec(params.elemsPerBeat, UInt(params.elemBits.W)))
    for (i <- 0 until params.elemsPerBeat) {
      elems(i) := data((i + 1) * params.elemBits - 1, i * params.elemBits)
    }
    elems(lane(elemLaneBits - 1, 0))
  }

  private val lastElem = row + 1.U === rows && col + 1.U === cols
  private val nextRow = Mux(col + 1.U === cols, row + 1.U, row)
  private val nextCol = Mux(col + 1.U === cols, 0.U, col + 1.U)
  private val nextElemAddr = elemAddr(nextRow, nextCol)
  private val nextElemInCachedBeat = alignedBeat(nextElemAddr) === alignedBeat(readElemAddr)

  io.cmd.ready := state === sIdle
  io.memReq.valid := state === sReq
  io.memReq.bits.addr := alignedBeat(elemAddr(row, col))
  io.memResp.ready := state === sWait
  io.out.valid := state === sOut
  io.out.bits.data := elemFromBeat(dataReg, elemLane(readElemAddr))
  io.out.bits.row := row
  io.out.bits.col := col
  io.out.bits.last := lastElem
  io.out.bits.error := errorReg
  io.active := state =/= sIdle
  io.bytesRead := Mux(io.memResp.fire, params.beatBytes.U(params.xLen.W), 0.U)

  when(io.cmd.fire) {
    base := io.cmd.bits.base
    rows := io.cmd.bits.rows
    cols := io.cmd.bits.cols
    stride := io.cmd.bits.stride
    row := 0.U
    col := 0.U
    errorReg := false.B
    state := Mux(io.cmd.bits.rows === 0.U || io.cmd.bits.cols === 0.U, sIdle, sReq)
  }

  when(io.memReq.fire) {
    readElemAddr := elemAddr(row, col)
    state := sWait
  }

  when(io.memResp.fire) {
    dataReg := io.memResp.bits.data
    errorReg := io.memResp.bits.error
    state := sOut
  }

  when(io.out.fire) {
    when(lastElem) {
      state := sIdle
    }.elsewhen(nextElemInCachedBeat) {
      row := nextRow
      col := nextCol
      readElemAddr := nextElemAddr
      state := sOut
    }.otherwise {
      row := nextRow
      col := nextCol
      state := sReq
    }
  }
}

class TileDmaWriter(params: AtikParams, tileRows: Int, tileCols: Int) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new TileDmaWriteCommand(params)))
    val tile = Input(Vec(tileRows, Vec(tileCols, UInt(params.elemBits.W))))
    val memReq = Decoupled(new DmaBeatRequest(params))
    val memData = Output(UInt(params.memDataBits.W))
    val memMask = Output(UInt((params.memDataBits / 8).W))
    val memResp = Flipped(Decoupled(new DmaWriteResponse))
    val active = Output(Bool())
    val done = Output(Bool())
    val error = Output(Bool())
    val bytesWritten = Output(UInt(params.xLen.W))
  })

  private val beatOffsetBits = log2Ceil(params.beatBytes)
  private val elemOffsetBits = log2Ceil(params.bytesPerElem)
  private val elemLaneBits = math.max(1, log2Ceil(params.elemsPerBeat))
  private val outColWideBits = math.max(1, log2Ceil(tileCols + params.elemsPerBeat + 1))
  private val outRowWideBits = math.max(1, log2Ceil(tileRows + 2))
  private val sIdle :: sWriteReq :: sWriteWait :: sDone :: sError :: Nil = Enum(5)
  private val state = RegInit(sIdle)

  private val base = Reg(UInt(params.addrBits.W))
  private val rows = Reg(UInt(32.W))
  private val cols = Reg(UInt(32.W))
  private val stride = Reg(UInt(32.W))
  private val outRow = RegInit(0.U(math.max(1, log2Ceil(tileRows)).W))
  private val outCol = RegInit(0.U(math.max(1, log2Ceil(tileCols)).W))
  private val errorReg = RegInit(false.B)

  private def elemOffset(index: UInt): UInt = (index << elemOffsetBits).pad(params.addrBits)
  private val elemAddr = base + elemOffset(outRow.pad(32) * stride + outCol.pad(32))
  private val alignedAddr = elemAddr & (~((params.beatBytes - 1).U(params.addrBits.W))).asUInt
  private val writeLane = elemAddr(beatOffsetBits - 1, elemOffsetBits)
  private val outColWide = outCol.pad(outColWideBits)
  private val outRowWide = outRow.pad(outRowWideBits)
  private val nextOutRowWide = outRowWide + 1.U(outRowWideBits.W)

  private val writeDataElems = Wire(Vec(params.elemsPerBeat, UInt(params.elemBits.W)))
  private val writeLaneValids = Wire(Vec(params.elemsPerBeat, Bool()))
  for (lane <- 0 until params.elemsPerBeat) {
    val laneU = lane.U(elemLaneBits.W)
    val laneAfterStart = laneU >= writeLane
    val laneDelta = Mux(laneAfterStart, (laneU - writeLane).pad(outColWideBits), 0.U(outColWideBits.W))
    val candidateCol = outColWide + laneDelta
    val valid = laneAfterStart && outRow.pad(32) < rows &&
      candidateCol < tileCols.U(outColWideBits.W) && candidateCol.pad(32) < cols
    val selected = Mux1H((0 until tileCols).map { col =>
      (candidateCol === col.U(outColWideBits.W)) -> io.tile(outRow)(col)
    })
    writeLaneValids(lane) := valid
    writeDataElems(lane) := Mux(valid, selected, 0.U)
  }

  private val writeByteMask = Wire(Vec(params.beatBytes, Bool()))
  for (byte <- 0 until params.beatBytes) {
    writeByteMask(byte) := writeLaneValids(byte / params.bytesPerElem)
  }
  private val writeElemsThisBeat = PopCount(writeLaneValids)
  private val nextOutColWide = outColWide + writeElemsThisBeat.pad(outColWideBits)
  private val writePayloadBytes = writeElemsThisBeat << elemOffsetBits
  private val onLastOutCol = nextOutColWide >= tileCols.U(outColWideBits.W) || nextOutColWide.pad(32) >= cols
  private val onLastOutRow = nextOutRowWide.pad(32) >= rows

  io.cmd.ready := state === sIdle
  io.memReq.valid := state === sWriteReq
  io.memReq.bits.addr := alignedAddr
  io.memData := Cat(writeDataElems.reverse)
  io.memMask := writeByteMask.asUInt
  io.memResp.ready := state === sWriteWait
  io.active := state =/= sIdle && state =/= sDone && state =/= sError
  io.done := state === sDone
  io.error := errorReg
  io.bytesWritten := Mux(io.memResp.fire, writePayloadBytes.pad(params.xLen), 0.U)

  when(io.cmd.fire) {
    base := io.cmd.bits.base
    rows := io.cmd.bits.rows
    cols := io.cmd.bits.cols
    stride := io.cmd.bits.stride
    outRow := 0.U
    outCol := 0.U
    errorReg := false.B
    state := Mux(io.cmd.bits.rows === 0.U || io.cmd.bits.cols === 0.U, sDone, sWriteReq)
  }

  when(io.memReq.fire) {
    state := sWriteWait
  }

  when(io.memResp.fire) {
    when(io.memResp.bits.error) {
      errorReg := true.B
      state := sError
    }.elsewhen(onLastOutCol && onLastOutRow) {
      state := sDone
    }.elsewhen(onLastOutCol) {
      outCol := 0.U
      outRow := nextOutRowWide(outRow.getWidth - 1, 0)
      state := sWriteReq
    }.otherwise {
      outCol := nextOutColWide(outCol.getWidth - 1, 0)
      state := sWriteReq
    }
  }

  when(state === sDone && !io.cmd.valid) {
    state := sIdle
  }
}
