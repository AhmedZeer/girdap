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

    val meshA = Output(Vec(params.meshRows, SInt(params.fixedBits.W)))
    val meshB = Output(Vec(params.meshCols, SInt(params.fixedBits.W)))
    val meshAccIn = Output(Vec(params.meshRows, Vec(params.meshCols, SInt(params.accumBits.W))))
    val meshOut = Input(Vec(params.meshRows, Vec(params.meshCols, SInt(params.accumBits.W))))
    val meshActive = Output(Bool())

    val busy = Output(Bool())
    val done = Output(Bool())
    val error = Output(UInt(8.W))
    val events = Output(new AtikCounterEvent(params))
  })

  private val elemOffsetBits = log2Ceil(params.bytesPerElem)
  private val rowIdxBits = math.max(1, log2Ceil(params.meshRows))
  private val colIdxBits = math.max(1, log2Ceil(params.meshCols))
  private val ktIdxBits = math.max(1, log2Ceil(params.matmulKt))

  private val (sIdle :: sInitTile :: sStartA :: sLoadA :: sStartB :: sLoadB ::
    sReadSram :: sLatchSram :: sCompute :: sNextChunkK :: sNextKChunk ::
    sStartWrite :: sWaitWrite :: sNextTile :: sDone :: sError :: Nil) = Enum(16)
  private val state = RegInit(sIdle)

  private val descReg = RegInit(0.U.asTypeOf(new AtikDescriptor(params)))
  private val errorReg = RegInit(AtikStatus.ok)
  private val tileM = RegInit(0.U(32.W))
  private val tileN = RegInit(0.U(32.W))
  private val kBase = RegInit(0.U(32.W))
  private val chunkK = RegInit(0.U(ktIdxBits.W))

  private val aRaw = RegInit(VecInit(Seq.fill(params.meshRows)(0.U(params.elemBits.W))))
  private val bRaw = RegInit(VecInit(Seq.fill(params.meshCols)(0.U(params.elemBits.W))))
  private val accum = RegInit(VecInit(Seq.fill(params.meshRows) {
    VecInit(Seq.fill(params.meshCols)(0.S(params.accumBits.W)))
  }))

  private val aSrams = Seq.fill(params.meshRows)(Module(new TileSram(params, params.matmulKt, params.elemBits)))
  private val bSrams = Seq.fill(params.meshCols)(Module(new TileSram(params, params.matmulKt, params.elemBits)))

  private def activeM(row: Int): Bool = tileM + row.U < descReg.m
  private def activeN(col: Int): Bool = tileN + col.U < descReg.n
  private def elemOffset(index: UInt): UInt = (index << elemOffsetBits).pad(params.addrBits)

  private val remainingM = descReg.m - tileM
  private val remainingN = descReg.n - tileN
  private val remainingK = descReg.k - kBase
  private val activeRows = Mux(remainingM > params.meshRows.U, params.meshRows.U(32.W), remainingM)
  private val activeCols = Mux(remainingN > params.meshCols.U, params.meshCols.U(32.W), remainingN)
  private val activeK = Mux(remainingK > params.matmulKt.U, params.matmulKt.U(32.W), remainingK)

  private val aTileBase = descReg.aAddr + elemOffset(tileM * descReg.lda + kBase)
  private val bTileBase = descReg.bAddr + elemOffset(kBase * descReg.ldb + tileN)
  private val outTileBase = descReg.outAddr + elemOffset(tileM * descReg.ldout + tileN)

  private val tileReader = Module(new TileDmaReader(params))
  private val tileWriter = Module(new TileDmaWriter(params, params.meshRows, params.meshCols))

  private val readCmd = Wire(new TileDmaReadCommand(params))
  readCmd.base := Mux(state === sStartA, aTileBase, bTileBase)
  readCmd.rows := Mux(state === sStartA, activeRows, activeK)
  readCmd.cols := Mux(state === sStartA, activeK, activeCols)
  readCmd.stride := Mux(state === sStartA, descReg.lda, descReg.ldb)

  tileReader.io.cmd.valid := state === sStartA || state === sStartB
  tileReader.io.cmd.bits := readCmd
  tileReader.io.out.ready := state === sLoadA || state === sLoadB

  io.memReadReq.valid := tileReader.io.memReq.valid
  io.memReadReq.bits := tileReader.io.memReq.bits
  tileReader.io.memReq.ready := io.memReadReq.ready
  tileReader.io.memResp.valid := io.memReadResp.valid
  tileReader.io.memResp.bits := io.memReadResp.bits
  io.memReadResp.ready := tileReader.io.memResp.ready

  private val writeCmd = Wire(new TileDmaWriteCommand(params))
  writeCmd.base := outTileBase
  writeCmd.rows := activeRows
  writeCmd.cols := activeCols
  writeCmd.stride := descReg.ldout

  private val outBf16 = Wire(Vec(params.meshRows, Vec(params.meshCols, UInt(params.elemBits.W))))
  tileWriter.io.cmd.valid := state === sStartWrite
  tileWriter.io.cmd.bits := writeCmd
  tileWriter.io.tile := outBf16

  io.memWriteReq.valid := tileWriter.io.memReq.valid
  io.memWriteReq.bits := tileWriter.io.memReq.bits
  tileWriter.io.memReq.ready := io.memWriteReq.ready
  io.memWriteData := tileWriter.io.memData
  io.memWriteMask := tileWriter.io.memMask
  tileWriter.io.memResp.valid := io.memWriteResp.valid
  tileWriter.io.memResp.bits := io.memWriteResp.bits
  io.memWriteResp.ready := tileWriter.io.memResp.ready

  for (r <- 0 until params.meshRows) {
    aSrams(r).io.wen := state === sLoadA && tileReader.io.out.fire && tileReader.io.out.bits.row === r.U
    aSrams(r).io.waddr := tileReader.io.out.bits.col(ktIdxBits - 1, 0)
    aSrams(r).io.wdata := tileReader.io.out.bits.data
    aSrams(r).io.raddr := chunkK
  }
  for (c <- 0 until params.meshCols) {
    bSrams(c).io.wen := state === sLoadB && tileReader.io.out.fire && tileReader.io.out.bits.col === c.U
    bSrams(c).io.waddr := tileReader.io.out.bits.row(ktIdxBits - 1, 0)
    bSrams(c).io.wdata := tileReader.io.out.bits.data
    bSrams(c).io.raddr := chunkK
  }

  private val aConverters = Seq.fill(params.meshRows)(Module(new Bf16ToFixed(params)))
  private val bConverters = Seq.fill(params.meshCols)(Module(new Bf16ToFixed(params)))
  for (r <- 0 until params.meshRows) {
    aConverters(r).io.in := Mux(activeM(r), aRaw(r), 0.U)
  }
  for (c <- 0 until params.meshCols) {
    bConverters(c).io.in := Mux(activeN(c), bRaw(c), 0.U)
  }

  for (r <- 0 until params.meshRows) {
    io.meshA(r) := aConverters(r).io.out
  }
  for (c <- 0 until params.meshCols) {
    io.meshB(c) := bConverters(c).io.out
  }
  for (r <- 0 until params.meshRows) {
    for (c <- 0 until params.meshCols) {
      io.meshAccIn(r)(c) := Mux(activeM(r) && activeN(c), accum(r)(c), 0.S)
    }
  }
  io.meshActive := state === sCompute

  private val outConverters = Seq.fill(params.meshRows, params.meshCols)(Module(new FixedToBf16(params, params.accumBits, params.accumFracBits)))
  for (r <- 0 until params.meshRows) {
    for (c <- 0 until params.meshCols) {
      outConverters(r)(c).io.in := accum(r)(c)
      outBf16(r)(c) := outConverters(r)(c).io.out
    }
  }

  private val badDims = io.desc.m === 0.U || io.desc.n === 0.U || io.desc.k === 0.U
  private val badAddr = io.desc.aAddr === 0.U || io.desc.bAddr === 0.U || io.desc.outAddr === 0.U
  private val onLastChunkK = chunkK.pad(32) + 1.U >= activeK
  private val onLastKChunk = kBase + activeK >= descReg.k
  private val onLastTileCol = tileN + params.meshCols.U >= descReg.n
  private val onLastTileRow = tileM + params.meshRows.U >= descReg.m

  private val event = WireDefault(0.U.asTypeOf(new AtikCounterEvent(params)))
  event.totalActive := state =/= sIdle && state =/= sDone && state =/= sError
  event.computeActive := state === sCompute
  event.meshActive := state === sCompute
  event.meshIdle := event.totalActive && state =/= sCompute
  event.softmaxActive := false.B
  event.dmaReadActive := tileReader.io.active
  event.dmaWriteActive := tileWriter.io.active
  event.dmaStall := (tileReader.io.memReq.valid && !tileReader.io.memReq.ready) ||
    (tileWriter.io.memReq.valid && !tileWriter.io.memReq.ready)
  event.sramStall := state === sReadSram
  event.bytesRead := tileReader.io.bytesRead
  event.bytesWritten := tileWriter.io.bytesWritten
  event.tilesLoaded := Mux(tileReader.io.out.fire && tileReader.io.out.bits.last, 1.U(params.xLen.W), 0.U(params.xLen.W))
  event.tilesComputed := Mux(state === sCompute, 1.U(params.xLen.W), 0.U(params.xLen.W))

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
      kBase := 0.U
      chunkK := 0.U
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
      kBase := 0.U
      chunkK := 0.U
      state := sStartA
    }
    is(sStartA) {
      when(tileReader.io.cmd.fire) {
        state := sLoadA
      }
    }
    is(sLoadA) {
      when(tileReader.io.out.fire) {
        when(tileReader.io.out.bits.error) {
          errorReg := AtikStatus.dma
          state := sError
        }.elsewhen(tileReader.io.out.bits.last) {
          state := sStartB
        }
      }
    }
    is(sStartB) {
      when(tileReader.io.cmd.fire) {
        state := sLoadB
      }
    }
    is(sLoadB) {
      when(tileReader.io.out.fire) {
        when(tileReader.io.out.bits.error) {
          errorReg := AtikStatus.dma
          state := sError
        }.elsewhen(tileReader.io.out.bits.last) {
          chunkK := 0.U
          state := sReadSram
        }
      }
    }
    is(sReadSram) {
      state := sLatchSram
    }
    is(sLatchSram) {
      for (r <- 0 until params.meshRows) {
        aRaw(r) := Mux(activeM(r), aSrams(r).io.rdata, 0.U)
      }
      for (c <- 0 until params.meshCols) {
        bRaw(c) := Mux(activeN(c), bSrams(c).io.rdata, 0.U)
      }
      state := sCompute
    }
    is(sCompute) {
      for (r <- 0 until params.meshRows) {
        for (c <- 0 until params.meshCols) {
          accum(r)(c) := io.meshOut(r)(c)
        }
      }
      state := sNextChunkK
    }
    is(sNextChunkK) {
      when(onLastChunkK) {
        state := sNextKChunk
      }.otherwise {
        chunkK := chunkK + 1.U
        state := sReadSram
      }
    }
    is(sNextKChunk) {
      when(onLastKChunk) {
        state := sStartWrite
      }.otherwise {
        kBase := kBase + activeK
        chunkK := 0.U
        for (r <- 0 until params.meshRows) {
          aRaw(r) := 0.U
        }
        for (c <- 0 until params.meshCols) {
          bRaw(c) := 0.U
        }
        state := sStartA
      }
    }
    is(sStartWrite) {
      when(tileWriter.io.cmd.fire) {
        state := sWaitWrite
      }
    }
    is(sWaitWrite) {
      when(tileWriter.io.error) {
        errorReg := AtikStatus.dma
        state := sError
      }.elsewhen(tileWriter.io.done) {
        state := sNextTile
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
