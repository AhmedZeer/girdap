package atik.control

import atik._
import atik.attention._
import atik.compute._
import atik.memory._
import atik.util.FixedPointUtil
import chisel3._
import chisel3.util._

class AttentionController(params: AtikParams) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val desc = Input(new AtikDescriptor(params))
    val causal = Input(Bool())

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

  private val (sIdle :: sInit :: sReqQ :: sWaitQ :: sReqK :: sWaitK ::
    sReadQkSram :: sLatchQkSram :: sComputeScore :: sNextDChunk ::
    sReqV :: sWaitV :: sInitPv :: sComputePv :: sNextPv :: sUpdateOnline ::
    sWriteReq :: sWriteWait :: sNextOut :: sDone :: sError :: Nil) = Enum(21)
  private val state = RegInit(sIdle)

  private val descReg = RegInit(0.U.asTypeOf(new AtikDescriptor(params)))
  private val causalReg = RegInit(false.B)
  private val errorReg = RegInit(AtikStatus.ok)
  private val qBase = RegInit(0.U(32.W))
  private val kvBase = RegInit(0.U(32.W))
  private val vColBase = RegInit(0.U(32.W))
  private val dIdx = RegInit(0.U(32.W))
  private val chunkD = RegInit(0.U(ktIdxBits.W))
  private val pvIdx = RegInit(0.U(colIdxBits.W))

  private val qRaw = RegInit(VecInit(Seq.fill(params.meshRows)(0.U(params.elemBits.W))))
  private val kRaw = RegInit(VecInit(Seq.fill(params.meshCols)(0.U(params.elemBits.W))))
  private val vRaw = RegInit(VecInit(Seq.fill(params.meshCols) {
    VecInit(Seq.fill(params.meshCols)(0.U(params.elemBits.W)))
  }))
  private val score = RegInit(VecInit(Seq.fill(params.meshRows) {
    VecInit(Seq.fill(params.meshCols)(0.S(params.accumBits.W)))
  }))
  private val qSrams = Seq.fill(params.meshRows)(Module(new TileSram(params, params.matmulKt, params.elemBits)))
  private val kSrams = Seq.fill(params.meshCols)(Module(new TileSram(params, params.matmulKt, params.elemBits)))
  private val rowMaxReg = RegInit(VecInit(Seq.fill(params.meshRows)(0.S(params.accumBits.W))))
  private val rowSumReg = RegInit(VecInit(Seq.fill(params.meshRows)(0.U(params.softmaxBits.W))))
  private val outAccReg = RegInit(VecInit(Seq.fill(params.meshRows) {
    VecInit(Seq.fill(params.meshCols)(0.S(params.accumBits.W)))
  }))

  private def elemOffset(index: UInt): UInt = (index << elemOffsetBits).pad(params.addrBits)
  private def qElemAddr(row: UInt): UInt = descReg.aAddr + elemOffset((qBase + row) * descReg.lda + dIdx)
  private def kElemAddr(col: UInt): UInt = descReg.bAddr + elemOffset((kvBase + col) * descReg.ldb + dIdx)
  private def vElemAddr(row: UInt, col: UInt): UInt = descReg.vAddr + elemOffset((kvBase + row) * descReg.ldv + vColBase + col)

  private val remainingD = descReg.k - dIdx
  private val activeD = Mux(remainingD > params.matmulKt.U, params.matmulKt.U(32.W), remainingD)
  private val activeRows = Mux(descReg.m - qBase > params.meshRows.U, params.meshRows.U(32.W), descReg.m - qBase)
  private val activeValueCols = Mux(descReg.n - vColBase > params.meshCols.U, params.meshCols.U(32.W), descReg.n - vColBase)
  private val outTileBase = descReg.outAddr + elemOffset(qBase * descReg.ldout + vColBase)
  private val tileReader = Module(new TileDmaReader(params))
  private val tileWriter = Module(new TileDmaWriter(params, params.meshRows, params.meshCols))

  private val qConverters = Seq.fill(params.meshRows)(Module(new Bf16ToFixed(params)))
  private val kConverters = Seq.fill(params.meshCols)(Module(new Bf16ToFixed(params)))
  private val vConverters = Seq.fill(params.meshCols, params.meshCols)(Module(new Bf16ToFixed(params)))
  private val scaleConverter = Module(new Bf16ToFixed(params))
  scaleConverter.io.in := descReg.scaleBf16

  for (r <- 0 until params.meshRows) {
    qConverters(r).io.in := qRaw(r)
  }
  for (kv <- 0 until params.meshCols) {
    kConverters(kv).io.in := kRaw(kv)
    for (vc <- 0 until params.meshCols) {
      vConverters(kv)(vc).io.in := vRaw(kv)(vc)
    }
  }

  for (r <- 0 until params.meshRows) {
    qSrams(r).io.wen := state === sWaitQ && tileReader.io.out.fire && tileReader.io.out.bits.row === r.U
    qSrams(r).io.waddr := tileReader.io.out.bits.col(ktIdxBits - 1, 0)
    qSrams(r).io.wdata := tileReader.io.out.bits.data
    qSrams(r).io.raddr := chunkD
  }
  for (kv <- 0 until params.meshCols) {
    kSrams(kv).io.wen := state === sWaitK && tileReader.io.out.fire && tileReader.io.out.bits.row === kv.U
    kSrams(kv).io.waddr := tileReader.io.out.bits.col(ktIdxBits - 1, 0)
    kSrams(kv).io.wdata := tileReader.io.out.bits.data
    kSrams(kv).io.raddr := chunkD
  }

  private val minScore = (-(BigInt(1) << (params.accumBits - 1))).S(params.accumBits.W)
  private val scaledScore = Wire(Vec(params.meshRows, Vec(params.meshCols, SInt(params.accumBits.W))))
  private val maskedScore = Wire(Vec(params.meshRows, Vec(params.meshCols, SInt(params.accumBits.W))))
  private val validScore = Wire(Vec(params.meshRows, Vec(params.meshCols, Bool())))
  private val tileRowMax = Wire(Vec(params.meshRows, SInt(params.accumBits.W)))
  private val newRowMax = Wire(Vec(params.meshRows, SInt(params.accumBits.W)))
  private val oldScale = Wire(Vec(params.meshRows, UInt(params.softmaxBits.W)))
  private val scoreExp = Wire(Vec(params.meshRows, Vec(params.meshCols, UInt(params.softmaxBits.W))))
  private val nextRowSum = Wire(Vec(params.meshRows, UInt(params.softmaxBits.W)))
  private val nextOutAcc = Wire(Vec(params.meshRows, Vec(params.meshCols, SInt(params.accumBits.W))))
  private val normalizedOut = Wire(Vec(params.meshRows, Vec(params.meshCols, SInt(params.accumBits.W))))
  private val outBf16 = Wire(Vec(params.meshRows, Vec(params.meshCols, UInt(params.elemBits.W))))

  private val scoreExpLuts = Seq.fill(params.meshRows, params.meshCols)(Module(new ExpLut(params, params.accumFracBits)))
  private val oldScaleLuts = Seq.fill(params.meshRows)(Module(new ExpLut(params, params.accumFracBits)))
  private val recipLuts = Seq.fill(params.meshRows)(Module(new ReciprocalLut(params)))
  private val outConverters = Seq.fill(params.meshRows, params.meshCols)(Module(new FixedToBf16(params, params.accumBits)))

  private val vFixed = Wire(Vec(params.meshCols, Vec(params.meshCols, SInt(params.fixedBits.W))))
  for (kv <- 0 until params.meshCols) {
    for (vc <- 0 until params.meshCols) {
      vFixed(kv)(vc) := vConverters(kv)(vc).io.out
    }
  }

  private def softToMeshFixed(value: UInt): SInt = {
    val shifted = if (params.softmaxFracBits >= params.fixedFracBits) {
      value >> (params.softmaxFracBits - params.fixedFracBits)
    } else {
      value << (params.fixedFracBits - params.softmaxFracBits)
    }
    FixedPointUtil.resizeSInt(shifted.asSInt, params.fixedBits)
  }

  private def fixedToMeshAccum(value: SInt): SInt = {
    FixedPointUtil.resizeSInt((value << params.fixedFracBits).asSInt, params.accumBits)
  }

  private val pvMeshActive = state === sComputePv
  for (r <- 0 until params.meshRows) {
    io.meshA(r) := Mux(
      pvMeshActive,
      softToMeshFixed(scoreExp(r)(pvIdx)),
      Mux(qBase + r.U < descReg.m, qConverters(r).io.out, 0.S)
    )
  }
  for (c <- 0 until params.meshCols) {
    io.meshB(c) := Mux(
      pvMeshActive,
      Mux(vColBase + c.U < descReg.n && kvBase + pvIdx < descReg.kvLen, vFixed(pvIdx)(c), 0.S),
      Mux(kvBase + c.U < descReg.kvLen, kConverters(c).io.out, 0.S)
    )
  }
  for (r <- 0 until params.meshRows) {
    for (c <- 0 until params.meshCols) {
      val validQkLane = qBase + r.U < descReg.m && kvBase + c.U < descReg.kvLen
      val validPvLane = qBase + r.U < descReg.m && vColBase + c.U < descReg.n
      io.meshAccIn(r)(c) := Mux(
        pvMeshActive,
        Mux(validPvLane, fixedToMeshAccum(outAccReg(r)(c)), 0.S),
        Mux(validQkLane, score(r)(c), 0.S)
      )
    }
  }
  io.meshActive := state === sComputeScore || pvMeshActive

  for (r <- 0 until params.meshRows) {
    for (kv <- 0 until params.meshCols) {
      val globalKv = kvBase + kv.U
      val globalQ = qBase + r.U
      val scaleProduct = score(r)(kv) * scaleConverter.io.out
      scaledScore(r)(kv) := FixedPointUtil.resizeSInt((scaleProduct >> params.fixedFracBits).asSInt, params.accumBits)
      validScore(r)(kv) := globalQ < descReg.m && globalKv < descReg.kvLen && (!causalReg || globalKv <= globalQ)
      maskedScore(r)(kv) := Mux(validScore(r)(kv), scaledScore(r)(kv), minScore)
    }

    tileRowMax(r) := (0 until params.meshCols).map(kv => maskedScore(r)(kv)).reduce { (a, b) => Mux(a > b, a, b) }
    newRowMax(r) := Mux(rowSumReg(r) === 0.U || tileRowMax(r) > rowMaxReg(r), tileRowMax(r), rowMaxReg(r))

    oldScaleLuts(r).io.in := Mux(
      rowSumReg(r) === 0.U,
      0.S(params.accumBits.W),
      FixedPointUtil.resizeSInt(rowMaxReg(r) - newRowMax(r), params.accumBits)
    )
    oldScale(r) := Mux(rowSumReg(r) === 0.U, 0.U, oldScaleLuts(r).io.out)

    for (kv <- 0 until params.meshCols) {
      scoreExpLuts(r)(kv).io.in := FixedPointUtil.resizeSInt(maskedScore(r)(kv) - newRowMax(r), params.accumBits)
      scoreExp(r)(kv) := Mux(validScore(r)(kv), scoreExpLuts(r)(kv).io.out, 0.U)
    }

    val rowSumWidth = params.softmaxBits + log2Ceil(params.meshCols + 2)
    val scaledOldSum = ((rowSumReg(r) * oldScale(r)) >> params.softmaxFracBits).asUInt
    val tileSum = (0 until params.meshCols).map(kv => scoreExp(r)(kv).pad(rowSumWidth)).reduce(_ + _)
    nextRowSum(r) := FixedPointUtil.resizeUInt(scaledOldSum.pad(rowSumWidth) + tileSum, params.softmaxBits)
    recipLuts(r).io.in := rowSumReg(r)

    for (vc <- 0 until params.meshCols) {
      val oldOutProduct = outAccReg(r)(vc) * oldScale(r).asSInt
      val scaledOldOut = (oldOutProduct >> params.softmaxFracBits).asSInt
      nextOutAcc(r)(vc) := FixedPointUtil.resizeSInt(scaledOldOut, params.accumBits)

      val normProduct = outAccReg(r)(vc) * recipLuts(r).io.out.asSInt
      normalizedOut(r)(vc) := FixedPointUtil.resizeSInt((normProduct >> params.softmaxFracBits).asSInt, params.accumBits)
      outConverters(r)(vc).io.in := Mux(qBase + r.U < descReg.m && vColBase + vc.U < descReg.n, normalizedOut(r)(vc), 0.S)
      outBf16(r)(vc) := outConverters(r)(vc).io.out
    }
  }

  private val writeCmd = Wire(new TileDmaWriteCommand(params))
  writeCmd.base := outTileBase
  writeCmd.rows := activeRows
  writeCmd.cols := activeValueCols
  writeCmd.stride := descReg.ldout
  tileWriter.io.cmd.valid := state === sWriteReq
  tileWriter.io.cmd.bits := writeCmd
  tileWriter.io.tile := outBf16

  private val readCmd = Wire(new TileDmaReadCommand(params))
  readCmd.base := MuxCase(qElemAddr(0.U), Seq(
    (state === sReqK) -> kElemAddr(0.U),
    (state === sReqV) -> vElemAddr(0.U, 0.U)
  ))
  readCmd.rows := MuxCase(activeRows, Seq(
    (state === sReqK) -> Mux(descReg.kvLen - kvBase > params.meshCols.U, params.meshCols.U(32.W), descReg.kvLen - kvBase),
    (state === sReqV) -> Mux(descReg.kvLen - kvBase > params.meshCols.U, params.meshCols.U(32.W), descReg.kvLen - kvBase)
  ))
  readCmd.cols := Mux(state === sReqV, activeValueCols, activeD)
  readCmd.stride := MuxCase(descReg.lda, Seq(
    (state === sReqK) -> descReg.ldb,
    (state === sReqV) -> descReg.ldv
  ))

  tileReader.io.cmd.valid := state === sReqQ || state === sReqK || state === sReqV
  tileReader.io.cmd.bits := readCmd
  tileReader.io.out.ready := state === sWaitQ || state === sWaitK || state === sWaitV
  io.memReadReq.valid := tileReader.io.memReq.valid
  io.memReadReq.bits := tileReader.io.memReq.bits
  tileReader.io.memReq.ready := io.memReadReq.ready
  tileReader.io.memResp.valid := io.memReadResp.valid
  tileReader.io.memResp.bits := io.memReadResp.bits
  io.memReadResp.ready := tileReader.io.memResp.ready
  io.memWriteReq.valid := tileWriter.io.memReq.valid
  io.memWriteReq.bits := tileWriter.io.memReq.bits
  tileWriter.io.memReq.ready := io.memWriteReq.ready
  io.memWriteData := tileWriter.io.memData
  io.memWriteMask := tileWriter.io.memMask
  tileWriter.io.memResp.valid := io.memWriteResp.valid
  tileWriter.io.memResp.bits := io.memWriteResp.bits
  io.memWriteResp.ready := tileWriter.io.memResp.ready

  private val badDims = io.desc.m === 0.U || io.desc.n === 0.U || io.desc.k === 0.U || io.desc.kvLen === 0.U
  private val badAddr = io.desc.aAddr === 0.U || io.desc.bAddr === 0.U || io.desc.vAddr === 0.U || io.desc.outAddr === 0.U
  private val unsupported = false.B
  private val nextQBase = qBase + params.meshRows.U(32.W)
  private val nextKvBase = kvBase + params.meshCols.U(32.W)
  private val nextVColBase = vColBase + params.meshCols.U(32.W)
  private val onLastChunkD = chunkD.pad(32) + 1.U >= activeD
  private val onLastDChunk = dIdx + activeD >= descReg.k
  private val onLastKvTile = nextKvBase >= descReg.kvLen
  private val onLastValueTile = nextVColBase >= descReg.n
  private val onLastQueryTile = nextQBase >= descReg.m

  private val event = WireDefault(0.U.asTypeOf(new AtikCounterEvent(params)))
  event.totalActive := state =/= sIdle && state =/= sDone && state =/= sError
  event.computeActive := state === sComputeScore || state === sInitPv || state === sComputePv || state === sUpdateOnline
  event.meshActive := state === sComputeScore || state === sComputePv
  event.meshIdle := event.totalActive && !(state === sComputeScore || state === sComputePv)
  event.softmaxActive := state === sInitPv || state === sUpdateOnline
  event.dmaReadActive := tileReader.io.active
  event.dmaWriteActive := tileWriter.io.active
  event.dmaStall := (tileReader.io.memReq.valid && !tileReader.io.memReq.ready) ||
    (tileWriter.io.memReq.valid && !tileWriter.io.memReq.ready)
  event.sramStall := state === sReadQkSram
  event.bytesRead := tileReader.io.bytesRead
  event.bytesWritten := tileWriter.io.bytesWritten
  event.tilesLoaded := Mux(tileReader.io.out.fire && tileReader.io.out.bits.last, 1.U(params.xLen.W), 0.U(params.xLen.W))
  event.tilesComputed := Mux(state === sComputeScore || state === sComputePv, 1.U(params.xLen.W), 0.U(params.xLen.W))

  when(io.start && state === sIdle) {
    descReg := io.desc
    causalReg := io.causal
    errorReg := AtikStatus.ok
    when(badDims) {
      errorReg := AtikStatus.badDims
      state := sError
    }.elsewhen(badAddr) {
      errorReg := AtikStatus.badAddr
      state := sError
    }.elsewhen(unsupported) {
      errorReg := AtikStatus.unsupportedConfig
      state := sError
    }.otherwise {
      state := sInit
    }
  }

  switch(state) {
    is(sInit) {
      qBase := 0.U
      kvBase := 0.U
      vColBase := 0.U
      dIdx := 0.U
      chunkD := 0.U
      pvIdx := 0.U
      for (r <- 0 until params.meshRows) {
        qRaw(r) := 0.U
        rowMaxReg(r) := minScore
        rowSumReg(r) := 0.U
        for (c <- 0 until params.meshCols) {
          score(r)(c) := 0.S
          outAccReg(r)(c) := 0.S
        }
      }
      for (c <- 0 until params.meshCols) {
        kRaw(c) := 0.U
        for (vc <- 0 until params.meshCols) {
          vRaw(c)(vc) := 0.U
        }
      }
      state := sReqQ
    }
    is(sReqQ) {
      when(tileReader.io.cmd.fire) {
        state := sWaitQ
      }
    }
    is(sWaitQ) {
      when(tileReader.io.out.fire) {
        when(tileReader.io.out.bits.error) {
          errorReg := AtikStatus.dma
          state := sError
        }.otherwise {
          when(tileReader.io.out.bits.last) {
            state := sReqK
          }
        }
      }
    }
    is(sReqK) {
      when(tileReader.io.cmd.fire) {
        state := sWaitK
      }
    }
    is(sWaitK) {
      when(tileReader.io.out.fire) {
        when(tileReader.io.out.bits.error) {
          errorReg := AtikStatus.dma
          state := sError
        }.otherwise {
          when(tileReader.io.out.bits.last) {
            chunkD := 0.U
            state := sReadQkSram
          }
        }
      }
    }
    is(sReadQkSram) {
      state := sLatchQkSram
    }
    is(sLatchQkSram) {
      for (r <- 0 until params.meshRows) {
        qRaw(r) := Mux(qBase + r.U < descReg.m, qSrams(r).io.rdata, 0.U)
      }
      for (kv <- 0 until params.meshCols) {
        kRaw(kv) := Mux(kvBase + kv.U < descReg.kvLen, kSrams(kv).io.rdata, 0.U)
      }
      state := sComputeScore
    }
    is(sComputeScore) {
      for (r <- 0 until params.meshRows) {
        for (kv <- 0 until params.meshCols) {
          when(qBase + r.U < descReg.m && kvBase + kv.U < descReg.kvLen) {
            score(r)(kv) := io.meshOut(r)(kv)
          }
        }
      }
      state := sNextDChunk
    }
    is(sNextDChunk) {
      when(onLastChunkD) {
        when(onLastDChunk) {
          state := sReqV
        }.otherwise {
          dIdx := dIdx + activeD
          chunkD := 0.U
          state := sReqQ
        }
      }.otherwise {
        chunkD := chunkD + 1.U
        state := sReadQkSram
      }
    }
    is(sReqV) {
      when(tileReader.io.cmd.fire) {
        state := sWaitV
      }
    }
    is(sWaitV) {
      when(tileReader.io.out.fire) {
        when(tileReader.io.out.bits.error) {
          errorReg := AtikStatus.dma
          state := sError
        }.otherwise {
          vRaw(tileReader.io.out.bits.row(colIdxBits - 1, 0))(tileReader.io.out.bits.col(colIdxBits - 1, 0)) := tileReader.io.out.bits.data
          when(tileReader.io.out.bits.last) {
            pvIdx := 0.U
            state := sInitPv
          }
        }
      }
    }
    is(sInitPv) {
      pvIdx := 0.U
      for (r <- 0 until params.meshRows) {
        for (vc <- 0 until params.meshCols) {
          outAccReg(r)(vc) := nextOutAcc(r)(vc)
        }
      }
      state := sComputePv
    }
    is(sComputePv) {
      for (r <- 0 until params.meshRows) {
        for (vc <- 0 until params.meshCols) {
          when(qBase + r.U < descReg.m && vColBase + vc.U < descReg.n) {
            outAccReg(r)(vc) := FixedPointUtil.resizeSInt((io.meshOut(r)(vc) >> params.fixedFracBits).asSInt, params.accumBits)
          }
        }
      }
      state := sNextPv
    }
    is(sNextPv) {
      when(pvIdx.pad(32) + 1.U >= params.meshCols.U || kvBase + pvIdx.pad(32) + 1.U >= descReg.kvLen) {
        state := sUpdateOnline
      }.otherwise {
        pvIdx := pvIdx + 1.U
        state := sComputePv
      }
    }
    is(sUpdateOnline) {
      for (r <- 0 until params.meshRows) {
        rowMaxReg(r) := newRowMax(r)
        rowSumReg(r) := nextRowSum(r)
      }
      when(onLastKvTile) {
        state := sWriteReq
      }.otherwise {
        kvBase := nextKvBase
        dIdx := 0.U
        chunkD := 0.U
        pvIdx := 0.U
        for (r <- 0 until params.meshRows) {
          for (kv <- 0 until params.meshCols) {
            score(r)(kv) := 0.S
          }
        }
        state := sReqQ
      }
    }
    is(sWriteReq) {
      when(tileWriter.io.cmd.fire) {
        state := sWriteWait
      }
    }
    is(sWriteWait) {
      when(tileWriter.io.error) {
        errorReg := AtikStatus.dma
        state := sError
      }.elsewhen(tileWriter.io.done) {
        state := sNextOut
      }
    }
    is(sNextOut) {
      when(onLastValueTile && onLastQueryTile) {
        state := sDone
      }.otherwise {
        when(onLastValueTile) {
          qBase := nextQBase
          vColBase := 0.U
        }.otherwise {
          vColBase := nextVColBase
        }
        kvBase := 0.U
        dIdx := 0.U
        chunkD := 0.U
        for (r <- 0 until params.meshRows) {
          qRaw(r) := 0.U
          rowMaxReg(r) := minScore
          rowSumReg(r) := 0.U
          for (c <- 0 until params.meshCols) {
            score(r)(c) := 0.S
            outAccReg(r)(c) := 0.S
          }
        }
        for (c <- 0 until params.meshCols) {
          kRaw(c) := 0.U
          for (vc <- 0 until params.meshCols) {
            vRaw(c)(vc) := 0.U
          }
        }
        state := sReqQ
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
