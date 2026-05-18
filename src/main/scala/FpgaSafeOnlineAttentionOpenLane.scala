package girdap

import chisel3._
import chisel3.util._
import scala.math.BigDecimal.RoundingMode

class FpgaSafeOnlineAttentionStandalone(
  private val nRows: Int,
  private val nCols: Int,
  private val maxK: Int,
  private val softmaxRecipLutEntries: Int,
  private val softmaxExpLutEntries: Int,
) extends Module {
  private val precision = 16
  private val fixedPointFracBits = 8
  private val accumPrec = 64
  private val softmaxIntPrecision = 32
  private val softmaxFracPrecision = 32
  private val softmaxExpLutRange = 16

  private val xLen = 64
  private val cacheDataBits = 128
  private val xlenBytes = xLen / 8
  private val beatBytes = cacheDataBits / 8
  private val beatOffsetBits = log2Ceil(beatBytes)
  private val wordsPerBeat = beatBytes / xlenBytes
  private val laneWidth = math.max(1, log2Ceil(wordsPerBeat))
  private val kWidth = log2Ceil(maxK + 1)
  private val rowCountWidth = log2Ceil(nRows + 1)
  private val colCountWidth = log2Ceil(nCols + 1)
  private val tileBaseWidth = log2Ceil(maxK + nCols + 1)
  private val scoreRowIdxWidth = log2Ceil(nRows)
  private val scoreTiles = maxK / nCols
  private val scoreTileIdxWidth = log2Ceil(scoreTiles)
  private val scoreBankAddrWidth = log2Ceil(nRows * scoreTiles)
  private val bufIdxWidth = log2Ceil(maxK)
  private val rowIdxWidth = log2Ceil(nRows)
  private val colIdxWidth = log2Ceil(nCols)
  private val outCount = nRows * nCols
  private val outputElemsPerWord = xLen / precision
  private val outputWordCount = (outCount + outputElemsPerWord - 1) / outputElemsPerWord
  private val outIdxWidth = log2Ceil(outputWordCount + 1)
  private val outWordIdxWidth = log2Ceil(outputWordCount)
  private val scoreFracBits = 2 * fixedPointFracBits
  private val pvProductShift = softmaxFracPrecision + fixedPointFracBits - scoreFracBits
  private val scoreScaleIntBits = 8
  private val softBitWidth = softmaxIntPrecision + softmaxFracPrecision
  private val lutEntries = softmaxRecipLutEntries
  private val lutBits = 64
  private val expLutEntries = softmaxExpLutEntries
  private val expLutRange = softmaxExpLutRange
  private val expLutIndexBits = log2Ceil(expLutEntries)
  private val expLutInputShift = scoreFracBits + log2Ceil(expLutRange) - expLutIndexBits
  private val minScoreFixed = (-(BigInt(1) << (accumPrec - 1))).S(accumPrec.W)

  require(pvProductShift >= 0)
  require(expLutInputShift >= 0)

  val io = IO(new Bundle {
    val cmdValid = Input(Bool())
    val cmdReady = Output(Bool())
    val cmdFunct = Input(UInt(7.W))
    val cmdRs1 = Input(UInt(xLen.W))
    val cmdRs2 = Input(UInt(xLen.W))
    val cmdRd = Input(UInt(5.W))

    val respValid = Output(Bool())
    val respReady = Input(Bool())
    val respRd = Output(UInt(5.W))
    val respData = Output(UInt(xLen.W))

    val memReqValid = Output(Bool())
    val memReqReady = Input(Bool())
    val memReqWrite = Output(Bool())
    val memReqAddr = Output(UInt(xLen.W))
    val memReqWdata = Output(UInt(cacheDataBits.W))
    val memReqMask = Output(UInt(beatBytes.W))

    val memRespValid = Input(Bool())
    val memRespReady = Output(Bool())
    val memRespData = Input(UInt(cacheDataBits.W))
    val memRespWriteAck = Input(Bool())

    val busy = Output(Bool())
  })

  private val qBuf = Reg(Vec(maxK, UInt(cacheDataBits.W)))
  private val kBuf = Reg(Vec(maxK, UInt(cacheDataBits.W)))
  private val vBuf = Reg(Vec(nCols, UInt(cacheDataBits.W)))
  private val scoreAccum = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.S(accumPrec.W))))))
  private val outAccum = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.S(accumPrec.W))))))
  private val rowMax = RegInit(VecInit(Seq.fill(nRows)(minScoreFixed)))
  private val rowDenom = RegInit(VecInit(Seq.fill(nRows)(0.U(softBitWidth.W))))
  private val scoreBanks = Seq.fill(nCols)(SyncReadMem(nRows * scoreTiles, SInt(accumPrec.W)))
  private val packedStoreWords = Reg(Vec(outputWordCount, UInt(xLen.W)))

  private val qBase = RegInit(0.U(xLen.W))
  private val kBase = RegInit(0.U(xLen.W))
  private val vBase = RegInit(0.U(xLen.W))
  private val outBase = RegInit(0.U(xLen.W))
  private val qRows = RegInit(0.U(rowCountWidth.W))
  private val kvRows = RegInit(0.U(kWidth.W))
  private val dK = RegInit(0.U(kWidth.W))
  private val valueCols = RegInit(0.U(colCountWidth.W))
  private val scaleBf16 = RegInit("h3F80".U(precision.W))
  private val qkConfigured = RegInit(false.B)
  private val voutConfigured = RegInit(false.B)
  private val dimsConfigured = RegInit(false.B)
  private val scoresReady = RegInit(false.B)
  private val applyAfterScores = RegInit(true.B)

  private val fillIdx = RegInit(0.U(kWidth.W))
  private val computeIdx = RegInit(0.U(kWidth.W))
  private val kvTileBase = RegInit(0.U(tileBaseWidth.W))
  private val activeKvCols = RegInit(0.U(colCountWidth.W))
  private val softRowIdx = RegInit(0.U(rowCountWidth.W))
  private val outIdx = RegInit(0.U(outIdxWidth.W))
  private val respRd = RegInit(0.U(5.W))
  private val respData = RegInit(0.U(xLen.W))

  private val states = Enum(21)
  private val sIdle = states(0)
  private val sReqFillQ = states(1)
  private val sWaitFillQ = states(2)
  private val sP1SetupTile = states(3)
  private val sReqFillK = states(4)
  private val sWaitFillK = states(5)
  private val sClearScores = states(6)
  private val sComputeScores = states(7)
  private val sP1RowLoad = states(8)
  private val sP1RowUpdate = states(9)
  private val sP2SetupTile = states(10)
  private val sReqFillV = states(11)
  private val sWaitFillV = states(12)
  private val sP2ScoreLatch = states(13)
  private val sP2RowLoad = states(14)
  private val sP2RowAccum = states(15)
  private val sQuantize = states(16)
  private val sReqPut = states(17)
  private val sWaitPut = states(18)
  private val sResp = states(19)
  private val sReqFillK2 = states(20)
  private val state = RegInit(sIdle)

  private def beatAddr(base: UInt, idx: UInt): UInt = base + idx.pad(xLen) * beatBytes.U
  private def kTileAddr(base: UInt, tileBase: UInt, idx: UInt): UInt =
    base + ((tileBase >> log2Ceil(nCols)).pad(xLen) * dK.pad(xLen) + idx.pad(xLen)) * beatBytes.U
  private def wordAddr(base: UInt, idx: UInt): UInt = base + idx.pad(xLen) * xlenBytes.U
  private def scoreBankAddr(row: UInt, tileBase: UInt): UInt = {
    val tileIdx = tileBase(tileBaseWidth - 1, log2Ceil(nCols))
    Cat(row(scoreRowIdxWidth - 1, 0), tileIdx(scoreTileIdxWidth - 1, 0))(scoreBankAddrWidth - 1, 0)
  }
  private def laneFromBeat(beat: UInt, lane: Int): UInt =
    beat(precision * (lane + 1) - 1, precision * lane)
  private def bufIdx(idx: UInt): UInt = idx(bufIdxWidth - 1, 0)
  private def rowIdx(idx: UInt): UInt = idx(rowIdxWidth - 1, 0)
  private def colIdx(idx: UInt): UInt = idx(colIdxWidth - 1, 0)
  private def outWordIdx(idx: UInt): UInt = idx(outWordIdxWidth - 1, 0)

  private val scoreReadAddr = scoreBankAddr(softRowIdx, kvTileBase)
  private val scoreReadData = Wire(Vec(nCols, SInt(accumPrec.W)))
  for (c <- 0 until nCols) {
    scoreReadData(c) := scoreBanks(c).read(scoreReadAddr, state === sP2RowLoad)
  }

  private val qLanes = Wire(Vec(nRows, UInt(precision.W)))
  private val kLanes = Wire(Vec(nCols, UInt(precision.W)))
  for (i <- 0 until nRows) {
    qLanes(i) := laneFromBeat(qBuf(bufIdx(computeIdx)), i)
    kLanes(i) := laneFromBeat(kBuf(bufIdx(computeIdx)), i)
  }

  private val qFixed = Seq.fill(nRows)(Module(new BFloat16ToSIntFixed(precision - fixedPointFracBits, fixedPointFracBits)))
  private val kFixed = Seq.fill(nCols)(Module(new BFloat16ToSIntFixed(precision - fixedPointFracBits, fixedPointFracBits)))
  for (i <- 0 until nRows) {
    qFixed(i).io.in := qLanes(i)
    kFixed(i).io.in := kLanes(i)
  }

  private val scaleConv = Module(new BFloat16ToSIntFixed(scoreScaleIntBits, scoreFracBits))
  scaleConv.io.in := scaleBf16
  private val scoreScaleFixed = scaleConv.io.out

  private val scoreFixed = Wire(Vec(nRows, Vec(nCols, SInt(accumPrec.W))))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      val scaledWide = (scoreAccum(r)(c) * scoreScaleFixed).asSInt
      val scaledShifted = (scaledWide >> scoreFracBits).asSInt
      scoreFixed(r)(c) := Mux(c.U < activeKvCols, scaledShifted(accumPrec - 1, 0).asSInt, minScoreFixed)
    }
  }

  private val softLatchedScores = Reg(Vec(nCols, SInt(accumPrec.W)))
  private val softVecFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  private val softProbFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  private val rowMaxRead = rowMax(rowIdx(softRowIdx))
  private val rowDenomRead = rowDenom(rowIdx(softRowIdx))

  private val softLutIndexBits = log2Ceil(lutEntries)
  private val softLut = VecInit(Seq.tabulate(lutEntries) { i =>
    val mant = BigDecimal(1) + (BigDecimal(i) + BigDecimal(0.5)) / BigDecimal(lutEntries)
    val recip = BigDecimal(1) / mant
    val scaled = (recip * BigDecimal(2).pow(lutBits - 1)).setScale(0, RoundingMode.HALF_UP).toBigInt
    scaled.U(lutBits.W)
  })
  private def reciprocalFixed(sum: UInt): UInt = {
    val nonZero = sum.orR
    val msb = (softBitWidth - 1).U - PriorityEncoder(Reverse(sum))
    val shift = Mux(nonZero, msb - softmaxFracPrecision.U, 0.U)
    val mant = sum >> shift
    val lutIndex = mant(softmaxFracPrecision - 1, softmaxFracPrecision - softLutIndexBits)
    val lutVal = softLut(lutIndex)
    val lutScaled = lutVal >> (lutBits - 1 - softmaxFracPrecision)
    val invRaw = lutScaled >> shift
    Mux(nonZero, invRaw.pad(softBitWidth)(softBitWidth - 1, 0), 0.U)
  }

  private val expLut = VecInit(Seq.tabulate(expLutEntries) { i =>
    val x = BigDecimal(i) * BigDecimal(expLutRange) / BigDecimal(math.max(expLutEntries - 1, 1))
    val scaled = (BigDecimal(math.exp(-x.toDouble)) * BigDecimal(2).pow(softmaxFracPrecision))
      .setScale(0, RoundingMode.HALF_UP).toBigInt
    scaled.U(softBitWidth.W)
  })
  private val expRangeFixed = (BigInt(expLutRange) << scoreFracBits).U((accumPrec + 1).W)
  private def expFixedLut(diff: SInt): UInt = {
    val diffWide = diff.pad(accumPrec + 1).asSInt
    val mag = Mux(diffWide >= 0.S, 0.U((accumPrec + 1).W), (-diffWide).asUInt)
    val rawIndex = (mag >> expLutInputShift)(expLutIndexBits - 1, 0)
    Mux(mag >= expRangeFixed, 0.U(softBitWidth.W), expLut(rawIndex))
  }

  private def fixedMax(a: SInt, b: SInt): SInt = Mux(a > b, a, b)
  private val softVecMaxFixed = softLatchedScores.reduceTree((a, b) => fixedMax(a, b))
  private val softGlobalMaxFixed = fixedMax(softVecMaxFixed, rowMaxRead)
  private val softMaxDiffFixed = rowMaxRead - softGlobalMaxFixed
  private val softMaxFixed = expFixedLut(softMaxDiffFixed)
  private val softOne = (1.U(softBitWidth.W) << softmaxFracPrecision)(softBitWidth - 1, 0)
  private val softMaxFixedClamped = Mux(softMaxFixed > softOne, softOne, softMaxFixed)
  private val softProdDenomFull = rowDenomRead * softMaxFixedClamped
  private val softProdDenom = (softProdDenomFull >> softmaxFracPrecision)(softBitWidth - 1, 0)
  private val softSubBase = Mux(state === sP2RowAccum, rowMaxRead, softGlobalMaxFixed)
  private val softInvSum = reciprocalFixed(rowDenomRead)

  for (i <- 0 until nCols) {
    val diff = softLatchedScores(i) - softSubBase
    softVecFixed(i) := expFixedLut(diff)
    val normFull = softVecFixed(i) * softInvSum
    softProbFixed(i) := (normFull >> softmaxFracPrecision)(softBitWidth - 1, 0)
  }
  private val softVecSum = softVecFixed.reduceTree(_ + _)
  private val softDenomNext = softVecSum + softProdDenom

  private val vFixed = Seq.fill(nCols, nCols)(Module(new BFloat16ToSIntFixed(precision - fixedPointFracBits, fixedPointFracBits)))
  for (kv <- 0 until nCols) {
    for (vc <- 0 until nCols) {
      vFixed(kv)(vc).io.in := laneFromBeat(vBuf(kv), vc)
    }
  }

  private val outBf16 = Wire(Vec(outCount, UInt(precision.W)))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      val conv = Module(new SIntFixedToBFloat16(accumPrec, scoreFracBits))
      conv.io.in := outAccum(r)(c)
      outBf16(r * nCols + c) := Mux(r.U < qRows && c.U < valueCols, conv.io.out, 0.U)
    }
  }

  private val packedStoreWordsWire = Wire(Vec(outputWordCount, UInt(xLen.W)))
  for (w <- 0 until outputWordCount) {
    val packedElems = Wire(Vec(outputElemsPerWord, UInt(precision.W)))
    for (e <- 0 until outputElemsPerWord) {
      val idx = w * outputElemsPerWord + e
      packedElems(e) := (if (idx < outCount) outBf16(idx) else 0.U)
    }
    packedStoreWordsWire(w) := packedElems.asUInt
  }

  private val writeAddr = wordAddr(outBase, outIdx)
  private val writeLane = writeAddr(beatOffsetBits - 1, log2Ceil(xlenBytes))
  private val writeBeatAddr = writeAddr & ~((beatBytes - 1).U(xLen.W))
  private val writeLaneWordIdxs = Wire(Vec(wordsPerBeat, UInt(outIdxWidth.W)))
  private val writeLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    val laneAfterStart = w.U(laneWidth.W) >= writeLane
    val laneDelta = Mux(laneAfterStart, w.U(laneWidth.W) - writeLane, 0.U(laneWidth.W))
    writeLaneWordIdxs(w) := outIdx + laneDelta
    writeLaneValids(w) := laneAfterStart && writeLaneWordIdxs(w) < outputWordCount.U
  }
  private val writeWordsThisBeat = PopCount(writeLaneValids)
  private val putDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    putDataWords(w) := Mux(writeLaneValids(w), packedStoreWords(outWordIdx(writeLaneWordIdxs(w))), 0.U)
  }
  private val putMaskBytes = Wire(Vec(beatBytes, Bool()))
  for (byte <- 0 until beatBytes) {
    putMaskBytes(byte) := writeLaneValids(byte / xlenBytes)
  }

  io.cmdReady := state === sIdle
  io.respValid := state === sResp
  io.respRd := respRd
  io.respData := respData
  io.busy := state =/= sIdle

  io.memReqValid := false.B
  io.memReqWrite := false.B
  io.memReqAddr := 0.U
  io.memReqWdata := 0.U
  io.memReqMask := 0.U
  io.memRespReady := false.B

  when(io.cmdValid && io.cmdReady) {
    respRd := io.cmdRd
    switch(io.cmdFunct) {
      is(8.U) {
        qBase := io.cmdRs1
        kBase := io.cmdRs2
        qkConfigured := true.B
        scoresReady := false.B
        respData := 0.U
        state := sResp
      }
      is(9.U) {
        vBase := io.cmdRs1
        outBase := io.cmdRs2
        voutConfigured := true.B
        respData := 0.U
        state := sResp
      }
      is(10.U) {
        val newQRows = io.cmdRs1(15, 0)(rowCountWidth - 1, 0)
        val newKvRows = io.cmdRs1(31, 16)(kWidth - 1, 0)
        val newDK = io.cmdRs1(47, 32)(kWidth - 1, 0)
        val newValueCols = io.cmdRs1(63, 48)(colCountWidth - 1, 0)
        when(newQRows =/= qRows || newKvRows =/= kvRows || newDK =/= dK) {
          scoresReady := false.B
        }
        qRows := newQRows
        kvRows := newKvRows
        dK := newDK
        valueCols := newValueCols
        dimsConfigured := true.B
        respData := 0.U
        state := sResp
      }
      is(11.U) {
        scaleBf16 := io.cmdRs1(precision - 1, 0)
        scoresReady := false.B
        respData := 0.U
        state := sResp
      }
      is(12.U, 13.U) {
        val dimsValid =
          qkConfigured && dimsConfigured &&
            (qRows =/= 0.U) && (qRows <= nRows.U) &&
            (kvRows =/= 0.U) && (kvRows <= maxK.U) &&
            (dK =/= 0.U) && (dK <= maxK.U) &&
            (io.cmdFunct === 13.U || (voutConfigured && valueCols =/= 0.U && valueCols <= nCols.U))
        respData := Mux(!qkConfigured || !dimsConfigured, 1.U, Mux(!dimsValid, 2.U, 0.U))
        when(dimsValid) {
          scoresReady := false.B
          applyAfterScores := io.cmdFunct === 12.U
          fillIdx := 0.U
          computeIdx := 0.U
          kvTileBase := 0.U
          activeKvCols := 0.U
          softRowIdx := 0.U
          outIdx := 0.U
          for (r <- 0 until nRows) {
            rowMax(r) := minScoreFixed
            rowDenom(r) := 0.U
            for (c <- 0 until nCols) {
              scoreAccum(r)(c) := 0.S
              outAccum(r)(c) := 0.S
            }
          }
          state := sReqFillQ
        }.otherwise {
          state := sResp
        }
      }
      is(14.U) {
        val dimsValid =
          voutConfigured && dimsConfigured && scoresReady &&
            (qRows =/= 0.U) && (qRows <= nRows.U) &&
            (kvRows =/= 0.U) && (kvRows <= maxK.U) &&
            (dK =/= 0.U) && (dK <= maxK.U) &&
            (valueCols =/= 0.U) && (valueCols <= nCols.U)
        respData := Mux(!voutConfigured || !dimsConfigured, 1.U, Mux(!scoresReady, 3.U, Mux(!dimsValid, 2.U, 0.U)))
        when(dimsValid) {
          fillIdx := 0.U
          computeIdx := 0.U
          kvTileBase := 0.U
          activeKvCols := 0.U
          softRowIdx := 0.U
          outIdx := 0.U
          for (r <- 0 until nRows) {
            for (c <- 0 until nCols) {
              outAccum(r)(c) := 0.S
            }
          }
          state := sP2SetupTile
        }.otherwise {
          state := sResp
        }
      }
      is(2.U) {
        respData := Cat(0.U((xLen - 2).W), state =/= sIdle, state === sResp)
        state := sResp
      }
      is(6.U) {
        respData := 0.U
        state := sResp
      }
    }
  }

  when(state === sReqFillQ) {
    io.memReqValid := fillIdx < dK
    io.memReqAddr := beatAddr(qBase, fillIdx)
    io.memReqMask := Fill(beatBytes, 1.U(1.W))
    when(io.memReqValid && io.memReqReady) {
      state := sWaitFillQ
    }
  }
  when(state === sWaitFillQ) {
    io.memRespReady := true.B
    when(io.memRespValid) {
      qBuf(bufIdx(fillIdx)) := io.memRespData
      val nextIdx = fillIdx + 1.U
      when(nextIdx >= dK) {
        fillIdx := 0.U
        kvTileBase := 0.U
        state := sP1SetupTile
      }.otherwise {
        fillIdx := nextIdx
        state := sReqFillQ
      }
    }
  }

  when(state === sP1SetupTile || state === sP2SetupTile) {
    val remaining = kvRows.pad(tileBaseWidth) - kvTileBase
    activeKvCols := Mux(remaining > nCols.U, nCols.U, remaining)(colCountWidth - 1, 0)
    fillIdx := 0.U
    when(state === sP2SetupTile) {
      softRowIdx := 0.U
      state := sReqFillV
    }.otherwise {
      state := sReqFillK
    }
  }

  when(state === sReqFillK || state === sReqFillK2) {
    io.memReqValid := fillIdx < dK
    io.memReqAddr := kTileAddr(kBase, kvTileBase, fillIdx)
    io.memReqMask := Fill(beatBytes, 1.U(1.W))
    when(io.memReqValid && io.memReqReady) {
      state := Mux(state === sReqFillK, sWaitFillK, sWaitFillK)
    }
  }
  when(state === sWaitFillK) {
    io.memRespReady := true.B
    when(io.memRespValid) {
      kBuf(bufIdx(fillIdx)) := io.memRespData
      val nextIdx = fillIdx + 1.U
      when(nextIdx >= dK) {
        fillIdx := 0.U
        computeIdx := 0.U
        state := sClearScores
      }.otherwise {
        fillIdx := nextIdx
        state := sReqFillK
      }
    }
  }

  when(state === sReqFillV) {
    io.memReqValid := fillIdx < activeKvCols
    io.memReqAddr := beatAddr(vBase, kvTileBase(kWidth - 1, 0) + fillIdx)
    io.memReqMask := Fill(beatBytes, 1.U(1.W))
    when(io.memReqValid && io.memReqReady) {
      state := sWaitFillV
    }
  }
  when(state === sWaitFillV) {
    io.memRespReady := true.B
    when(io.memRespValid) {
      vBuf(colIdx(fillIdx)) := io.memRespData
      val nextIdx = fillIdx + 1.U
      when(nextIdx >= activeKvCols) {
        fillIdx := 0.U
        computeIdx := 0.U
        state := sP2RowLoad
      }.otherwise {
        fillIdx := nextIdx
        state := sReqFillV
      }
    }
  }

  when(state === sClearScores) {
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        scoreAccum(r)(c) := 0.S
      }
    }
    computeIdx := 0.U
    state := sComputeScores
  }
  when(state === sComputeScores) {
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        val product = qFixed(r).io.out * kFixed(c).io.out
        val sum = scoreAccum(r)(c) + product.asSInt
        scoreAccum(r)(c) := sum(accumPrec - 1, 0).asSInt
      }
    }
    when(computeIdx + 1.U >= dK) {
      softRowIdx := 0.U
      state := sP1RowLoad
    }.otherwise {
      computeIdx := computeIdx + 1.U
    }
  }
  when(state === sP1RowLoad) {
    for (c <- 0 until nCols) {
      softLatchedScores(c) := scoreFixed(rowIdx(softRowIdx))(c)
    }
    state := sP1RowUpdate
  }
  when(state === sP1RowUpdate) {
    when(softRowIdx < qRows) {
      val writeScoreAddr = scoreBankAddr(softRowIdx, kvTileBase)
      for (c <- 0 until nCols) {
        val scoreIdx = kvTileBase + c.U
        when(c.U < activeKvCols && scoreIdx < kvRows) {
          scoreBanks(c).write(writeScoreAddr, softLatchedScores(c))
        }
      }
      rowMax(rowIdx(softRowIdx)) := softGlobalMaxFixed
      rowDenom(rowIdx(softRowIdx)) := softDenomNext
    }
    when(softRowIdx + 1.U >= qRows) {
      val nextTile = kvTileBase + nCols.U
      when(nextTile >= kvRows) {
        kvTileBase := 0.U
        scoresReady := true.B
        when(applyAfterScores) {
          state := sP2SetupTile
        }.otherwise {
          respData := 0.U
          state := sResp
        }
      }.otherwise {
        kvTileBase := nextTile
        state := sP1SetupTile
      }
    }.otherwise {
      softRowIdx := softRowIdx + 1.U
      state := sP1RowLoad
    }
  }

  when(state === sP2RowLoad) {
    state := sP2ScoreLatch
  }
  when(state === sP2ScoreLatch) {
    for (c <- 0 until nCols) {
      softLatchedScores(c) := Mux(c.U < activeKvCols, scoreReadData(c), minScoreFixed)
    }
    state := sP2RowAccum
  }
  when(state === sP2RowAccum) {
    when(softRowIdx < qRows) {
      for (vc <- 0 until nCols) {
        val terms = Seq.tabulate(nCols) { kv =>
          val product = (softProbFixed(kv).asSInt * vFixed(kv)(vc).io.out).asSInt
          val shiftedProduct = (product >> pvProductShift).asSInt
          Mux(kv.U < activeKvCols && vc.U < valueCols, shiftedProduct.pad(accumPrec).asSInt, 0.S(accumPrec.W))
        }
        val sum = outAccum(rowIdx(softRowIdx))(vc) + terms.reduce(_ + _)
        outAccum(rowIdx(softRowIdx))(vc) := sum(accumPrec - 1, 0).asSInt
      }
    }
    when(softRowIdx + 1.U >= qRows) {
      val nextTile = kvTileBase + nCols.U
      when(nextTile >= kvRows) {
        outIdx := 0.U
        state := sQuantize
      }.otherwise {
        kvTileBase := nextTile
        state := sP2SetupTile
      }
    }.otherwise {
      softRowIdx := softRowIdx + 1.U
      state := sP2RowLoad
    }
  }

  when(state === sQuantize) {
    for (w <- 0 until outputWordCount) {
      packedStoreWords(w) := packedStoreWordsWire(w)
    }
    outIdx := 0.U
    state := sReqPut
  }
  when(state === sReqPut) {
    io.memReqValid := outIdx < outputWordCount.U
    io.memReqWrite := true.B
    io.memReqAddr := writeBeatAddr
    io.memReqWdata := putDataWords.asUInt
    io.memReqMask := putMaskBytes.asUInt
    when(io.memReqValid && io.memReqReady) {
      state := sWaitPut
    }
  }
  when(state === sWaitPut) {
    io.memRespReady := true.B
    when(io.memRespValid) {
      val nextOutIdx = outIdx + writeWordsThisBeat
      when(nextOutIdx >= outputWordCount.U) {
        respData := 0.U
        state := sResp
      }.otherwise {
        outIdx := nextOutIdx
        state := sReqPut
      }
    }
  }

  when(io.respValid && io.respReady) {
    state := sIdle
  }
}

class FpgaSafeOnlineAttention8x8Standalone
    extends FpgaSafeOnlineAttentionStandalone(
      nRows = 8,
      nCols = 8,
      maxK = 256,
      softmaxRecipLutEntries = 256,
      softmaxExpLutEntries = 512,
    )

class FpgaSafeOnlineAttention4x4Standalone
    extends FpgaSafeOnlineAttentionStandalone(
      nRows = 4,
      nCols = 4,
      maxK = 128,
      softmaxRecipLutEntries = 64,
      softmaxExpLutEntries = 128,
    )

class FpgaSafeOnlineAttention8x8OpenLaneTop extends Module {
  val io = IO(new Bundle {
    val cmdValid = Input(Bool())
    val cmdReady = Output(Bool())
    val cmdFunct = Input(UInt(7.W))
    val cmdRs1 = Input(UInt(64.W))
    val cmdRs2 = Input(UInt(64.W))
    val cmdRd = Input(UInt(5.W))
    val respValid = Output(Bool())
    val respReady = Input(Bool())
    val respRd = Output(UInt(5.W))
    val respData = Output(UInt(64.W))
    val memReqValid = Output(Bool())
    val memReqReady = Input(Bool())
    val memReqWrite = Output(Bool())
    val memReqAddr = Output(UInt(64.W))
    val memReqWdata = Output(UInt(128.W))
    val memReqMask = Output(UInt(16.W))
    val memRespValid = Input(Bool())
    val memRespReady = Output(Bool())
    val memRespData = Input(UInt(128.W))
    val memRespWriteAck = Input(Bool())
    val busy = Output(Bool())
  })

  val dut = Module(new FpgaSafeOnlineAttention8x8Standalone)
  dut.io <> io
}

class FpgaSafeOnlineAttention4x4OpenLaneTop extends Module {
  val io = IO(new Bundle {
    val cmdValid = Input(Bool())
    val cmdReady = Output(Bool())
    val cmdFunct = Input(UInt(7.W))
    val cmdRs1 = Input(UInt(64.W))
    val cmdRs2 = Input(UInt(64.W))
    val cmdRd = Input(UInt(5.W))
    val respValid = Output(Bool())
    val respReady = Input(Bool())
    val respRd = Output(UInt(5.W))
    val respData = Output(UInt(64.W))
    val memReqValid = Output(Bool())
    val memReqReady = Input(Bool())
    val memReqWrite = Output(Bool())
    val memReqAddr = Output(UInt(64.W))
    val memReqWdata = Output(UInt(128.W))
    val memReqMask = Output(UInt(16.W))
    val memRespValid = Input(Bool())
    val memRespReady = Output(Bool())
    val memRespData = Input(UInt(128.W))
    val memRespWriteAck = Input(Bool())
    val busy = Output(Bool())
  })

  val dut = Module(new FpgaSafeOnlineAttention4x4Standalone)
  dut.io <> io
}

object EmitFpgaSafeOnlineAttention8x8OpenLane extends App {
  val targetDir =
    if (args.nonEmpty) args(0)
    else "/home/ubuntu/firesim_build_disk/openlane_attention/src"

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new FpgaSafeOnlineAttention8x8OpenLaneTop,
    Array("--target-dir", targetDir),
    Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    )
  )
}

object EmitFpgaSafeOnlineAttention4x4OpenLane extends App {
  val targetDir =
    if (args.nonEmpty) args(0)
    else "/home/ubuntu/firesim_build_disk/openlane_attention_4x4/src"

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new FpgaSafeOnlineAttention4x4OpenLaneTop,
    Array("--target-dir", targetDir),
    Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    )
  )
}
