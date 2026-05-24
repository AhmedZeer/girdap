package atik.control

import atik._
import atik.compute._
import atik.memory._
import chisel3._
import chisel3.util._

class MatmulController(params: AtikParams) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val desc = Input(new AtikDescriptor(params))

    val memReadReq = Decoupled(new DmaBeatRequest(params))
    val memReadResp = Flipped(Decoupled(new DmaBeatResponse(params)))
    val memWriteReq = Decoupled(new DmaBeatRequest(params))
    val memWriteData = Output(UInt(params.memDataBits.W))
    val memWriteMask = Output(UInt((params.memDataBits / 8).W))
    val memWriteResp = Flipped(Decoupled(new DmaWriteResponse))

    val busy = Output(Bool())
    val done = Output(Bool())
    val error = Output(UInt(8.W))
    val events = Output(new AtikCounterEvent(params))
  })

  private val beatOffsetBits = log2Ceil(params.beatBytes)
  private val elemOffsetBits = log2Ceil(params.bytesPerElem)
  private val elemLaneBits = math.max(1, log2Ceil(params.elemsPerBeat))
  private val rowIdxBits = math.max(1, log2Ceil(params.meshRows))
  private val colIdxBits = math.max(1, log2Ceil(params.meshCols))

  private val (sIdle :: sInitTile :: sReqA :: sWaitA :: sReqB :: sWaitB ::
    sCompute :: sNextK :: sWriteReq :: sWriteWait :: sNextOut :: sNextTile :: sDone :: sError :: Nil) = Enum(14)
  private val state = RegInit(sIdle)

  private val descReg = RegInit(0.U.asTypeOf(new AtikDescriptor(params)))
  private val errorReg = RegInit(AtikStatus.ok)
  private val tileM = RegInit(0.U(32.W))
  private val tileN = RegInit(0.U(32.W))
  private val kIdx = RegInit(0.U(32.W))
  private val loadAIdx = RegInit(0.U(rowIdxBits.W))
  private val loadBIdx = RegInit(0.U(colIdxBits.W))
  private val outRow = RegInit(0.U(rowIdxBits.W))
  private val outCol = RegInit(0.U(colIdxBits.W))
  private val readElemAddr = RegInit(0.U(params.addrBits.W))
  private val writeElemAddr = Wire(UInt(params.addrBits.W))

  private val aRaw = RegInit(VecInit(Seq.fill(params.meshRows)(0.U(params.elemBits.W))))
  private val bRaw = RegInit(VecInit(Seq.fill(params.meshCols)(0.U(params.elemBits.W))))
  private val accum = RegInit(VecInit(Seq.fill(params.meshRows) {
    VecInit(Seq.fill(params.meshCols)(0.S(params.accumBits.W)))
  }))

  private def activeM(row: Int): Bool = tileM + row.U < descReg.m
  private def activeN(col: Int): Bool = tileN + col.U < descReg.n
  private def alignedBeat(addr: UInt): UInt = addr & (~((params.beatBytes - 1).U(params.addrBits.W))).asUInt
  private def elemLane(addr: UInt): UInt = addr(beatOffsetBits - 1, elemOffsetBits)
  private def elemFromBeat(data: UInt, lane: UInt): UInt = {
    val elems = Wire(Vec(params.elemsPerBeat, UInt(params.elemBits.W)))
    for (i <- 0 until params.elemsPerBeat) {
      elems(i) := data((i + 1) * params.elemBits - 1, i * params.elemBits)
    }
    elems(lane(elemLaneBits - 1, 0))
  }
  private def aElemAddr(row: UInt): UInt = {
    descReg.aAddr + (((tileM + row) * descReg.lda + kIdx) << elemOffsetBits)(params.addrBits - 1, 0)
  }
  private def bElemAddr(col: UInt): UInt = {
    descReg.bAddr + ((kIdx * descReg.ldb + tileN + col) << elemOffsetBits)(params.addrBits - 1, 0)
  }

  private val aConverters = Seq.fill(params.meshRows)(Module(new Bf16ToFixed(params)))
  private val bConverters = Seq.fill(params.meshCols)(Module(new Bf16ToFixed(params)))
  for (r <- 0 until params.meshRows) {
    aConverters(r).io.in := Mux(activeM(r), aRaw(r), 0.U)
  }
  for (c <- 0 until params.meshCols) {
    bConverters(c).io.in := Mux(activeN(c), bRaw(c), 0.U)
  }

  private val mesh = Module(new MacMesh(params))
  for (r <- 0 until params.meshRows) {
    mesh.io.a(r) := aConverters(r).io.out
    for (c <- 0 until params.meshCols) {
      mesh.io.b(c) := bConverters(c).io.out
      mesh.io.accIn(r)(c) := Mux(activeM(r) && activeN(c), accum(r)(c), 0.S)
    }
  }

  private val outConverters = Seq.fill(params.meshRows, params.meshCols)(Module(new FixedToBf16(params, params.accumBits)))
  private val outBf16 = Wire(Vec(params.meshRows, Vec(params.meshCols, UInt(params.elemBits.W))))
  for (r <- 0 until params.meshRows) {
    for (c <- 0 until params.meshCols) {
      outConverters(r)(c).io.in := accum(r)(c)
      outBf16(r)(c) := outConverters(r)(c).io.out
    }
  }

  private val activeLoadA = tileM + loadAIdx < descReg.m
  private val activeLoadB = tileN + loadBIdx < descReg.n
  writeElemAddr := descReg.outAddr + (((tileM + outRow) * descReg.ldout + tileN + outCol) << elemOffsetBits)(params.addrBits - 1, 0)

  private val writeLane = elemLane(writeElemAddr)
  private val outColWideBits = math.max(1, log2Ceil(params.meshCols + params.elemsPerBeat + 1))
  private val outColWide = outCol.pad(outColWideBits)
  private val rowActiveForWrite = tileM + outRow < descReg.m
  private val writeDataElems = Wire(Vec(params.elemsPerBeat, UInt(params.elemBits.W)))
  private val writeLaneValids = Wire(Vec(params.elemsPerBeat, Bool()))

  for (lane <- 0 until params.elemsPerBeat) {
    val laneU = lane.U(elemLaneBits.W)
    val laneAfterStart = laneU >= writeLane
    val laneDelta = Mux(laneAfterStart, (laneU - writeLane).pad(outColWideBits), 0.U(outColWideBits.W))
    val candidateCol = outColWide + laneDelta
    val globalCol = tileN + candidateCol
    val inMeshCol = candidateCol < params.meshCols.U(outColWideBits.W)
    val inMatrixCol = globalCol < descReg.n
    val valid = rowActiveForWrite && laneAfterStart && inMeshCol && inMatrixCol
    val selected = Mux1H((0 until params.meshCols).map { col =>
      (candidateCol === col.U(outColWideBits.W)) -> outBf16(outRow)(col)
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

  io.memReadReq.valid := (state === sReqA && activeLoadA) || (state === sReqB && activeLoadB)
  io.memReadReq.bits.addr := alignedBeat(Mux(state === sReqA, aElemAddr(loadAIdx), bElemAddr(loadBIdx)))
  io.memReadResp.ready := state === sWaitA || state === sWaitB
  io.memWriteReq.valid := state === sWriteReq
  io.memWriteReq.bits.addr := alignedBeat(writeElemAddr)
  io.memWriteData := Cat(writeDataElems.reverse)
  io.memWriteMask := writeByteMask.asUInt
  io.memWriteResp.ready := state === sWriteWait

  private val badDims = io.desc.m === 0.U || io.desc.n === 0.U || io.desc.k === 0.U
  private val badAddr = io.desc.aAddr === 0.U || io.desc.bAddr === 0.U || io.desc.outAddr === 0.U
  private val onLastK = kIdx + 1.U >= descReg.k
  private val onLastOutCol = nextOutColWide >= params.meshCols.U(outColWideBits.W) || tileN + nextOutColWide >= descReg.n
  private val onLastOutRow = outRow + 1.U >= params.meshRows.U || tileM + outRow + 1.U >= descReg.m
  private val onLastTileCol = tileN + params.meshCols.U >= descReg.n
  private val onLastTileRow = tileM + params.meshRows.U >= descReg.m

  private val event = WireDefault(0.U.asTypeOf(new AtikCounterEvent(params)))
  event.totalActive := state =/= sIdle && state =/= sDone && state =/= sError
  event.computeActive := state === sCompute
  event.meshActive := state === sCompute
  event.meshIdle := event.totalActive && state =/= sCompute
  event.dmaReadActive := state === sReqA || state === sWaitA || state === sReqB || state === sWaitB
  event.dmaWriteActive := state === sWriteReq || state === sWriteWait
  event.bytesRead := Mux(io.memReadResp.fire, params.beatBytes.U(params.xLen.W), 0.U)
  event.bytesWritten := Mux(io.memWriteResp.fire, writePayloadBytes.pad(params.xLen), 0.U)

  when(io.start && state === sIdle) {
    descReg := io.desc
    errorReg := AtikStatus.ok
    when(badDims) {
      errorReg := AtikStatus.badDims
      state := sError
    }.elsewhen(badAddr) {
      errorReg := AtikStatus.badAddr
      state := sError
    }.otherwise {
      tileM := 0.U
      tileN := 0.U
      kIdx := 0.U
      state := sInitTile
    }
  }

  switch(state) {
    is(sInitTile) {
      for (r <- 0 until params.meshRows) {
        aRaw(r) := 0.U
        for (c <- 0 until params.meshCols) {
          accum(r)(c) := 0.S
        }
      }
      for (c <- 0 until params.meshCols) {
        bRaw(c) := 0.U
      }
      kIdx := 0.U
      loadAIdx := 0.U
      loadBIdx := 0.U
      outRow := 0.U
      outCol := 0.U
      state := sReqA
    }
    is(sReqA) {
      when(!activeLoadA) {
        aRaw(loadAIdx) := 0.U
        when(loadAIdx === (params.meshRows - 1).U) {
          loadBIdx := 0.U
          state := sReqB
        }.otherwise {
          loadAIdx := loadAIdx + 1.U
        }
      }.elsewhen(io.memReadReq.fire) {
        readElemAddr := aElemAddr(loadAIdx)
        state := sWaitA
      }
    }
    is(sWaitA) {
      when(io.memReadResp.fire) {
        when(io.memReadResp.bits.error) {
          errorReg := AtikStatus.dma
          state := sError
        }.otherwise {
          aRaw(loadAIdx) := elemFromBeat(io.memReadResp.bits.data, elemLane(readElemAddr))
          when(loadAIdx === (params.meshRows - 1).U) {
            loadBIdx := 0.U
            state := sReqB
          }.otherwise {
            loadAIdx := loadAIdx + 1.U
            state := sReqA
          }
        }
      }
    }
    is(sReqB) {
      when(!activeLoadB) {
        bRaw(loadBIdx) := 0.U
        when(loadBIdx === (params.meshCols - 1).U) {
          state := sCompute
        }.otherwise {
          loadBIdx := loadBIdx + 1.U
        }
      }.elsewhen(io.memReadReq.fire) {
        readElemAddr := bElemAddr(loadBIdx)
        state := sWaitB
      }
    }
    is(sWaitB) {
      when(io.memReadResp.fire) {
        when(io.memReadResp.bits.error) {
          errorReg := AtikStatus.dma
          state := sError
        }.otherwise {
          bRaw(loadBIdx) := elemFromBeat(io.memReadResp.bits.data, elemLane(readElemAddr))
          when(loadBIdx === (params.meshCols - 1).U) {
            state := sCompute
          }.otherwise {
            loadBIdx := loadBIdx + 1.U
            state := sReqB
          }
        }
      }
    }
    is(sCompute) {
      for (r <- 0 until params.meshRows) {
        for (c <- 0 until params.meshCols) {
          accum(r)(c) := mesh.io.out(r)(c)
        }
      }
      state := sNextK
    }
    is(sNextK) {
      when(onLastK) {
        outRow := 0.U
        outCol := 0.U
        state := sWriteReq
      }.otherwise {
        kIdx := kIdx + 1.U
        loadAIdx := 0.U
        loadBIdx := 0.U
        state := sReqA
      }
    }
    is(sWriteReq) {
      when(io.memWriteReq.fire) {
        state := sWriteWait
      }
    }
    is(sWriteWait) {
      when(io.memWriteResp.fire) {
        when(io.memWriteResp.bits.error) {
          errorReg := AtikStatus.dma
          state := sError
        }.otherwise {
          state := sNextOut
        }
      }
    }
    is(sNextOut) {
      when(onLastOutCol && onLastOutRow) {
        state := sNextTile
      }.elsewhen(onLastOutCol) {
        outCol := 0.U
        outRow := outRow + 1.U
        state := sWriteReq
      }.otherwise {
        outCol := nextOutColWide(colIdxBits - 1, 0)
        state := sWriteReq
      }
    }
    is(sNextTile) {
      when(onLastTileCol && onLastTileRow) {
        state := sDone
      }.elsewhen(onLastTileCol) {
        tileN := 0.U
        tileM := tileM + params.meshRows.U
        state := sInitTile
      }.otherwise {
        tileN := tileN + params.meshCols.U
        state := sInitTile
      }
    }
    is(sDone) {
      when(!io.start) {
        state := sIdle
      }
    }
  }

  io.busy := state =/= sIdle && state =/= sDone && state =/= sError
  io.done := state === sDone
  io.error := errorReg
  io.events := event
}
