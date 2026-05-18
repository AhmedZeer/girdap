package girdap

import chisel3._
import chisel3.util._

import freechips.rocketchip.diplomacy.IdRange
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import org.chipsalliance.cde.config.Parameters

class FusedAttentionWSImpl(
  outer: FusedAttentionWSRoCC
)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters
  with HasL1CacheParameters {

  val cacheParams = tileParams.dcache.get

  private val precision = outer.precision
  private val nRows = outer.nRows
  private val nCols = outer.nCols
  private val maxK = outer.maxK
  private val fixedPointFracBits = outer.fixedPointFracBits
  private val accumPrec = outer.accumBits
  private val softmaxIntPrecision = outer.softmaxIntPrecision
  private val softmaxFracPrecision = outer.softmaxFracPrecision
  private val tlSourceIds = outer.numTLSourceIds

  private val xlenBytes = xLen / 8
  private val lgXlenBytes = log2Ceil(xlenBytes)
  private val beatBytes = cacheDataBits / 8
  private val beatOffsetBits = log2Ceil(beatBytes)
  private val wordsPerBeat = beatBytes / xlenBytes
  private val laneWidth = math.max(1, log2Ceil(wordsPerBeat))
  private val cacheIdxWidth = log2Ceil(maxK + 1)
  private val cacheVecIdxWidth = math.max(1, log2Ceil(maxK))
  private val chunkWidth = log2Ceil(nRows + 1)
  private val rowCountWidth = log2Ceil(nRows + 1)
  private val rowVecIdxWidth = math.max(1, log2Ceil(nRows))
  private val colCountWidth = log2Ceil(nCols + 1)
  private val tileShift = log2Ceil(nCols)
  private val scoreFracBits = 2 * fixedPointFracBits
  private val outputElemsPerWord = xLen / precision
  private val outCount = nRows * nCols
  private val outputWordCount = (outCount + outputElemsPerWord - 1) / outputElemsPerWord
  private val outIdxWidth = log2Ceil(outputWordCount + 1)
  private val outWordIdxWidth = math.max(1, log2Ceil(outputWordCount))
  private val readTlSourceId = 0
  private val writeTlSourceId = tlSourceIds - 1
  private val softBitWidth = softmaxIntPrecision + softmaxFracPrecision
  private val lutEntries = 256
  private val lutBits = 64
  private val minBf16 = "hFF80".U(16.W)
  private val scoreScaleIntBits = 8
  private val scoreScaleWidth = scoreScaleIntBits + scoreFracBits
  private val perfCounterWidth = log2Ceil(13)

  require(nRows == nCols, "fused attention WS core requires a square mesh")
  require(isPow2(nCols), s"fused attention expects a power-of-two tile width, got $nCols")
  require(cacheDataBits % xLen == 0, s"cacheDataBits must be a multiple of xLen ($cacheDataBits % $xLen)")
  require(wordsPerBeat > 0 && isPow2(wordsPerBeat), s"wordsPerBeat must be a positive power of two, got $wordsPerBeat")
  require(precision * nRows <= xLen, s"A-stream word exceeds xLen (${precision * nRows} > $xLen)")
  require(precision * nCols <= xLen, s"B-stream word exceeds xLen (${precision * nCols} > $xLen)")
  require(tlSourceIds >= 2, s"numTLSourceIds must be >= 2, got $tlSourceIds")
  require(softmaxFracPrecision >= log2Ceil(lutEntries) - 1, "softmax frac precision too small for LUT indexing")

  private val perfBusyCycles = 0
  private val perfRunCmds = 1
  private val perfTLReadReqs = 2
  private val perfTLWriteReqs = 3
  private val perfWaitQFillCycles = 4
  private val perfWaitKFillCycles = 5
  private val perfQKComputeCycles = 6
  private val perfScorePostCycles = 7
  private val perfSoftmaxPass1Cycles = 8
  private val perfSoftmaxPass2Cycles = 9
  private val perfWaitVFillCycles = 10
  private val perfPVComputeCycles = 11
  private val perfWaitPutCycles = 12
  private val numPerfCounters = 13
  private val perfCounters = RegInit(VecInit(Seq.fill(numPerfCounters)(0.U(xLen.W))))

  private val states = Enum(25)
  private val s_idle = states(0)
  private val s_req_fill_q = states(1)
  private val s_wait_fill_q = states(2)
  private val s_qk_setup_tile = states(3)
  private val s_req_fill_b = states(4)
  private val s_wait_fill_b = states(5)
  private val s_qk_prep_chunk = states(6)
  private val s_qk_load_weights = states(7)
  private val s_qk_feed_rows = states(8)
  private val s_qk_wait_chunk_out = states(9)
  private val s_qk_store_scores = states(10)
  private val s_softmax_row_init = states(11)
  private val s_softmax_p1_load = states(12)
  private val s_softmax_p1_compute = states(13)
  private val s_softmax_p2_load = states(14)
  private val s_softmax_p2_compute = states(15)
  private val s_pv_setup = states(16)
  private val s_pv_prep_chunk = states(17)
  private val s_pv_load_weights = states(18)
  private val s_pv_feed_rows = states(19)
  private val s_pv_wait_chunk_out = states(20)
  private val s_quantize_out = states(21)
  private val s_req_put = states(22)
  private val s_wait_put = states(23)
  private val s_resp = states(24)
  private val state = RegInit(s_idle)

  private val core = Module(new WeightStationaryArrayCoreBF16Fixed(
    precision = precision,
    nRows = nRows,
    nCols = nCols,
    accumPrec = accumPrec,
    fracBits = fixedPointFracBits
  ))

  private val aTileBuf = RegInit(VecInit(Seq.fill(maxK)(VecInit(Seq.fill(nRows)(0.U(precision.W))))))
  private val bTileBuf = RegInit(VecInit(Seq.fill(maxK)(VecInit(Seq.fill(nCols)(0.U(precision.W))))))
  private val scoreBuf = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(maxK)(0.U(precision.W))))))
  private val accumTile = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.U(accumPrec.W))))))

  private val flatOut = Wire(Vec(outCount, UInt(accumPrec.W)))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      flatOut(r * nCols + c) := accumTile(r)(c)
    }
  }

  private val packedBFloatOut = Wire(Vec(outCount, UInt(precision.W)))
  private val packedStoreWordsWire = Wire(Vec(outputWordCount, UInt(xLen.W)))
  private val packedStoreWordsReg = Reg(Vec(outputWordCount, UInt(xLen.W)))

  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      val conv = Module(new SIntFixedToBFloat16(accumPrec, scoreFracBits))
      conv.io.in := accumTile(r)(c).asSInt
      packedBFloatOut(r * nCols + c) := conv.io.out
    }
  }

  for (w <- 0 until outputWordCount) {
    val packedElems = Wire(Vec(outputElemsPerWord, UInt(precision.W)))
    for (e <- 0 until outputElemsPerWord) {
      val idx = w * outputElemsPerWord + e
      if (idx < outCount) {
        packedElems(e) := packedBFloatOut(idx)
      } else {
        packedElems(e) := 0.U
      }
    }
    packedStoreWordsWire(w) := packedElems.asUInt
  }

  private val qBase = RegInit(0.U(xLen.W))
  private val kTilesBase = RegInit(0.U(xLen.W))
  private val vBase = RegInit(0.U(xLen.W))
  private val outBase = RegInit(0.U(xLen.W))
  private val qRows = RegInit(0.U(rowCountWidth.W))
  private val kvRows = RegInit(0.U(cacheIdxWidth.W))
  private val dK = RegInit(0.U(cacheIdxWidth.W))
  private val valueCols = RegInit(0.U(colCountWidth.W))
  private val scaleBf16 = RegInit("h3F80".U(16.W))
  private val causalEn = RegInit(false.B)
  private val queryPosBase = RegInit(0.U(xLen.W))
  private val qkAddrsConfigured = RegInit(false.B)
  private val vOutAddrsConfigured = RegInit(false.B)
  private val dimsConfigured = RegInit(false.B)
  private val addrsConfigured = qkAddrsConfigured && vOutAddrsConfigured

  private val qFillIdx = RegInit(0.U(cacheIdxWidth.W))
  private val bFillIdx = RegInit(0.U(cacheIdxWidth.W))
  private val bFillBaseAddr = RegInit(0.U(xLen.W))
  private val bFillCount = RegInit(0.U(cacheIdxWidth.W))
  private val bFillForPV = RegInit(false.B)

  private val activeTileCols = RegInit(0.U(colCountWidth.W))
  private val qkTileBase = RegInit(0.U(cacheIdxWidth.W))

  private val computeKBaseIdx = RegInit(0.U(cacheIdxWidth.W))
  private val computeKRemaining = RegInit(0.U(cacheIdxWidth.W))
  private val computeChunkK = RegInit(0.U(chunkWidth.W))
  private val feedRowIdx = RegInit(0.U(chunkWidth.W))
  private val captureRowIdx = RegInit(0.U(chunkWidth.W))

  private val softRowIdx = RegInit(0.U(rowCountWidth.W))
  private val softColIdx = RegInit(0.U(cacheIdxWidth.W))
  private val softAccum = RegInit(0.U(softBitWidth.W))
  private val softMaxPrev = RegInit(minBf16)
  private val softInvSum = RegInit(0.U(softBitWidth.W))
  private val softLatchedScores = Reg(Vec(nCols, UInt(precision.W)))

  private val outIdx = RegInit(0.U(outIdxWidth.W))
  private val respRd = RegInit(0.U(5.W))
  private val respData = RegInit(0.U(xLen.W))

  private val weightTile = Wire(Vec(nRows, Vec(nCols, UInt(precision.W))))
  private val currentInVec = Wire(Vec(nRows, UInt(precision.W)))

  for (wr <- 0 until nRows) {
    val weightIdx = (computeKBaseIdx + wr.U)(cacheVecIdxWidth - 1, 0)
    for (wc <- 0 until nCols) {
      weightTile(wr)(wc) := Mux(
        (wr.U < computeChunkK) && (wc.U < activeTileCols),
        bTileBuf(weightIdx)(wc),
        0.U
      )
    }
  }

  for (lane <- 0 until nRows) {
    val srcWordIdx = (computeKBaseIdx + lane.U)(cacheVecIdxWidth - 1, 0)
    val rowValid = feedRowIdx < qRows
    val rowIdx = feedRowIdx(rowVecIdxWidth - 1, 0)
    currentInVec(lane) := Mux(
      (lane.U < computeChunkK) && rowValid,
      aTileBuf(srcWordIdx)(rowIdx),
      0.U
    )
  }

  core.io.loadWeights := false.B
  core.io.clearWeights := false.B
  core.io.weightTile := weightTile
  core.io.inValid := false.B
  core.io.inVec := currentInVec

  private val scoreScaleConv = Module(new BFloat16ToSIntFixed(scoreScaleIntBits, scoreFracBits))
  scoreScaleConv.io.in := scaleBf16
  private val scoreScaleFixed = scoreScaleConv.io.out
  private val scorePostBf16 = Wire(Vec(nRows, Vec(nCols, UInt(precision.W))))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      val scaledWide = (accumTile(r)(c).asSInt * scoreScaleFixed).asSInt
      val scaledShifted = (scaledWide >> scoreFracBits).asSInt
      val scaledTrimmed = scaledShifted(accumPrec - 1, 0).asSInt
      val conv = Module(new SIntFixedToBFloat16(accumPrec, scoreFracBits))
      conv.io.in := scaledTrimmed
      val causalCutoff = queryPosBase + r.U(xLen.W)
      val causalCol = qkTileBase.pad(xLen) + c.U(xLen.W)
      val masked = causalEn && (causalCol > causalCutoff)
      scorePostBf16(r)(c) := Mux(masked, minBf16, conv.io.out)
    }
  }

  private val softVecMax = Module(new BFloat16VectorMax(nCols))
  private val softGlobalMax = Module(new BFloat16Max)
  private val softMaxSub = Module(new BFloat16Sub)
  private val softMaxExp = Module(new BFloat16Exp)
  private val softMaxFixedP = Module(new BFloat16ToFixed(softmaxIntPrecision, softmaxFracPrecision))
  private val softVecSubs = Seq.fill(nCols)(Module(new BFloat16Sub))
  private val softVecExps = Seq.fill(nCols)(Module(new BFloat16Exp))
  private val softVecFixedPs = Seq.fill(nCols)(Module(new BFloat16ToFixed(softmaxIntPrecision, softmaxFracPrecision)))
  private val softVecNormOut = Seq.fill(nCols)(Module(new UIntFixedToBFloat16(softBitWidth, softmaxFracPrecision)))
  private val softVecFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  private val softVecNormFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  private val softProbBf16 = Wire(Vec(nCols, UInt(precision.W)))

  softVecMax.io.in := softLatchedScores.asUInt
  softGlobalMax.io.in_1 := softVecMax.io.out
  softGlobalMax.io.in_2 := softMaxPrev
  private val softSubBase = Mux(state === s_softmax_p2_compute, softMaxPrev, softGlobalMax.io.out)

  softMaxSub.io.in_1 := softMaxPrev
  softMaxSub.io.in_2 := softGlobalMax.io.out
  softMaxExp.io.in := softMaxSub.io.out
  softMaxFixedP.io.in := softMaxExp.io.out

  private val softOne = (1.U(softBitWidth.W) << softmaxFracPrecision)(softBitWidth - 1, 0)
  private val softMaxFixedPClamped = Mux(softMaxFixedP.io.out > softOne, softOne, softMaxFixedP.io.out)
  private val softProdAccumFull = softAccum * softMaxFixedPClamped
  private val softProdAccum = (softProdAccumFull >> softmaxFracPrecision)(softBitWidth - 1, 0)

  for (i <- 0 until nCols) {
    softVecSubs(i).io.in_1 := softLatchedScores(i)
    softVecSubs(i).io.in_2 := softSubBase
    softVecExps(i).io.in := softVecSubs(i).io.out
    softVecFixedPs(i).io.in := softVecExps(i).io.out
    softVecFixed(i) := softVecFixedPs(i).io.out

    val normFull = softVecFixed(i) * softInvSum
    softVecNormFixed(i) := (normFull >> softmaxFracPrecision)(softBitWidth - 1, 0)
    softVecNormOut(i).io.in := softVecNormFixed(i)
    softProbBf16(i) := softVecNormOut(i).io.out
  }

  private val softVecSum = softVecFixed.reduceTree(_ + _)
  private val softAccumNext = softVecSum + softProdAccum
  private val softLutIndexBits = log2Ceil(lutEntries)
  private val softLut = VecInit(Seq.tabulate(lutEntries) { i =>
    val mant = BigDecimal(1) + (BigDecimal(i) + BigDecimal(0.5)) / BigDecimal(lutEntries)
    val recip = BigDecimal(1) / mant
    val scaled = (recip * BigDecimal(2).pow(lutBits - 1)).setScale(0, BigDecimal.RoundingMode.HALF_UP).toBigInt
    scaled.U(lutBits.W)
  })
  private val softSumNonZero = softAccumNext.orR
  private val softMsb = (softBitWidth - 1).U - PriorityEncoder(Reverse(softAccumNext))
  private val softShift = Mux(softSumNonZero, softMsb - softmaxFracPrecision.U, 0.U)
  private val softMant = softAccumNext >> softShift
  private val softLutIndex = softMant(softmaxFracPrecision, softmaxFracPrecision - softLutIndexBits + 1)
  private val softLutVal = softLut(softLutIndex)
  private val softScaleDown = lutBits - 1 - softmaxFracPrecision
  private val softLutScaled =
    if (softScaleDown >= 0) (softLutVal >> softScaleDown) else (softLutVal << (-softScaleDown))
  private val softInvRaw = softLutScaled >> softShift
  private val softInvSumCandidate = Mux(softSumNonZero, softInvRaw.pad(softBitWidth)(softBitWidth - 1, 0), 0.U)

  private val (tlOut, edgesOut) = outer.atlNode.out(0)
  private val dHasData = edgesOut.hasData(tlOut.d.bits)
  private val dSource = tlOut.d.bits.source(math.max(1, log2Ceil(tlSourceIds)) - 1, 0)
  private val dIsReadResp = dHasData && (tlOut.d.bits.opcode === TLMessages.AccessAckData)
  private val dIsWriteAck = !dHasData && (tlOut.d.bits.opcode === TLMessages.AccessAck)

  private val readDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    readDataWords(w) := tlOut.d.bits.data((w + 1) * xLen - 1, w * xLen)
  }

  private val qReadAddr = qBase + qFillIdx * xlenBytes.U
  private val qReadBeatAddr = if (wordsPerBeat > 1) qReadAddr & ~((beatBytes - 1).U(xLen.W)) else qReadAddr
  private val qReadLane = if (wordsPerBeat > 1) qReadAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  private val qReadDestIdxs = Wire(Vec(wordsPerBeat, UInt(cacheIdxWidth.W)))
  private val qReadLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    if (wordsPerBeat > 1) {
      val laneAfterStart = w.U(laneWidth.W) >= qReadLane
      val laneDelta = Mux(laneAfterStart, w.U(laneWidth.W) - qReadLane, 0.U(laneWidth.W))
      qReadDestIdxs(w) := qFillIdx + laneDelta
      qReadLaneValids(w) := laneAfterStart && (qReadDestIdxs(w) < dK)
    } else {
      qReadDestIdxs(w) := qFillIdx
      qReadLaneValids(w) := qFillIdx < dK
    }
  }
  private val qWordsThisBeat = PopCount(qReadLaneValids)
  private val qGetBits = edgesOut.Get(
    fromSource = readTlSourceId.U,
    toAddress = qReadBeatAddr,
    lgSize = (if (wordsPerBeat > 1) beatOffsetBits.U else lgXlenBytes.U)
  )._2

  private val bReadAddr = bFillBaseAddr + bFillIdx * xlenBytes.U
  private val bReadBeatAddr = if (wordsPerBeat > 1) bReadAddr & ~((beatBytes - 1).U(xLen.W)) else bReadAddr
  private val bReadLane = if (wordsPerBeat > 1) bReadAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  private val bReadDestIdxs = Wire(Vec(wordsPerBeat, UInt(cacheIdxWidth.W)))
  private val bReadLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    if (wordsPerBeat > 1) {
      val laneAfterStart = w.U(laneWidth.W) >= bReadLane
      val laneDelta = Mux(laneAfterStart, w.U(laneWidth.W) - bReadLane, 0.U(laneWidth.W))
      bReadDestIdxs(w) := bFillIdx + laneDelta
      bReadLaneValids(w) := laneAfterStart && (bReadDestIdxs(w) < bFillCount)
    } else {
      bReadDestIdxs(w) := bFillIdx
      bReadLaneValids(w) := bFillIdx < bFillCount
    }
  }
  private val bWordsThisBeat = PopCount(bReadLaneValids)
  private val bGetBits = edgesOut.Get(
    fromSource = readTlSourceId.U,
    toAddress = bReadBeatAddr,
    lgSize = (if (wordsPerBeat > 1) beatOffsetBits.U else lgXlenBytes.U)
  )._2

  private val writeAddr = outBase + outIdx * xlenBytes.U
  private val writeLane = if (wordsPerBeat > 1) writeAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  private val writeBeatAddr = if (wordsPerBeat > 1) writeAddr & ~((beatBytes - 1).U(xLen.W)) else writeAddr
  private val writeLaneWordIdxs = Wire(Vec(wordsPerBeat, UInt(outIdxWidth.W)))
  private val writeLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    if (wordsPerBeat > 1) {
      val laneAfterStart = w.U(laneWidth.W) >= writeLane
      val laneDelta = Mux(laneAfterStart, w.U(laneWidth.W) - writeLane, 0.U(laneWidth.W))
      writeLaneWordIdxs(w) := outIdx + laneDelta
      writeLaneValids(w) := laneAfterStart && (writeLaneWordIdxs(w) < outputWordCount.U)
    } else {
      writeLaneWordIdxs(w) := outIdx
      writeLaneValids(w) := outIdx < outputWordCount.U
    }
  }
  private val writeWordsThisBeat = PopCount(writeLaneValids)
  private val putDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    val outWordIdx = writeLaneWordIdxs(w)(outWordIdxWidth - 1, 0)
    putDataWords(w) := Mux(writeLaneValids(w), packedStoreWordsReg(outWordIdx), 0.U)
  }
  private val putMaskBytes = Wire(Vec(beatBytes, Bool()))
  for (byte <- 0 until beatBytes) {
    putMaskBytes(byte) := writeLaneValids(byte / xlenBytes)
  }
  private val putBits = edgesOut.Put(
    fromSource = writeTlSourceId.U,
    toAddress = writeBeatAddr,
    lgSize = (if (wordsPerBeat > 1) beatOffsetBits.U else lgXlenBytes.U),
    data = putDataWords.asUInt,
    mask = putMaskBytes.asUInt
  )._2

  tlOut.a.valid := false.B
  tlOut.a.bits := qGetBits
  tlOut.d.ready := false.B

  io.busy := state =/= s_idle
  io.interrupt := false.B
  io.cmd.ready := state === s_idle
  io.resp.valid := state === s_resp
  io.resp.bits.rd := respRd
  io.resp.bits.data := respData

  private val clearPerfCounters = WireDefault(false.B)
  private val incRunCmd = WireDefault(false.B)
  private val incTLReadReq = WireDefault(false.B)
  private val incTLWriteReq = WireDefault(false.B)

  when(io.cmd.fire) {
    respRd := io.cmd.bits.inst.rd
    val funct = io.cmd.bits.inst.funct

    when(funct === 0.U) {
      qBase := io.cmd.bits.rs1
      kTilesBase := io.cmd.bits.rs2
      qkAddrsConfigured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 1.U) {
      vBase := io.cmd.bits.rs1
      outBase := io.cmd.bits.rs2
      vOutAddrsConfigured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 2.U) {
      qRows := io.cmd.bits.rs1(15, 0)(rowCountWidth - 1, 0)
      kvRows := io.cmd.bits.rs1(31, 16)(cacheIdxWidth - 1, 0)
      dK := io.cmd.bits.rs1(47, 32)(cacheIdxWidth - 1, 0)
      valueCols := io.cmd.bits.rs1(63, 48)(colCountWidth - 1, 0)
      dimsConfigured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 3.U) {
      scaleBf16 := io.cmd.bits.rs1(15, 0)
      causalEn := io.cmd.bits.rs1(16)
      queryPosBase := io.cmd.bits.rs2
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 4.U) {
      incRunCmd := true.B
      val dimsValid =
        addrsConfigured &&
          dimsConfigured &&
          (qRows =/= 0.U) &&
          (kvRows =/= 0.U) &&
          (dK =/= 0.U) &&
          (valueCols =/= 0.U) &&
          (qRows <= nRows.U) &&
          (kvRows <= maxK.U) &&
          (dK <= maxK.U) &&
          (valueCols <= nCols.U)
      respData := Mux(!addrsConfigured || !dimsConfigured, 1.U,
        Mux(!dimsValid, 2.U, 0.U))
      when(dimsValid) {
        qFillIdx := 0.U
        bFillIdx := 0.U
        qkTileBase := 0.U
        computeKBaseIdx := 0.U
        computeKRemaining := 0.U
        computeChunkK := 0.U
        feedRowIdx := 0.U
        captureRowIdx := 0.U
        activeTileCols := 0.U
        softRowIdx := 0.U
        softColIdx := 0.U
        softAccum := 0.U
        softMaxPrev := minBf16
        softInvSum := 0.U
        outIdx := 0.U
        for (k <- 0 until maxK) {
          for (r <- 0 until nRows) {
            aTileBuf(k)(r) := 0.U
          }
        }
        for (r <- 0 until nRows) {
          for (c <- 0 until nCols) {
            accumTile(r)(c) := 0.U
          }
        }
        state := s_req_fill_q
      }.otherwise {
        state := s_resp
      }
    }.elsewhen(funct === 5.U) {
      respData := Cat(
        0.U((xLen - 4).W),
        state =/= s_idle,
        dimsConfigured,
        addrsConfigured,
        state === s_resp
      )
      state := s_resp
    }.elsewhen(funct === 6.U) {
      val counterSel = io.cmd.bits.rs1
      val counterIdx = counterSel(perfCounterWidth - 1, 0)
      respData := Mux(counterSel < numPerfCounters.U, perfCounters(counterIdx), "hBAD0BAD0BAD0BAD0".U)
      state := s_resp
    }.elsewhen(funct === 7.U) {
      clearPerfCounters := true.B
      respData := 0.U
      state := s_resp
    }.otherwise {
      respData := "hDEAD".U
      state := s_resp
    }
  }

  when(core.io.outValid) {
    when(captureRowIdx < qRows) {
      val captureIdx = captureRowIdx(rowVecIdxWidth - 1, 0)
      for (c <- 0 until nCols) {
        when(c.U < activeTileCols) {
          accumTile(captureIdx)(c) := accumTile(captureIdx)(c) + core.io.outVec(c)
        }
      }
    }
    captureRowIdx := captureRowIdx + 1.U
  }

  when(state === s_req_fill_q) {
    tlOut.a.valid := true.B
    tlOut.a.bits := qGetBits
    when(tlOut.a.fire) {
      incTLReadReq := true.B
      state := s_wait_fill_q
    }
  }

  when(state === s_wait_fill_q) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp, "fused attention expected AccessAckData on packed-Q fill")
      assert(dSource === readTlSourceId.U, "fused attention expected Q fill response on read source")
      for (w <- 0 until wordsPerBeat) {
        when(qReadLaneValids(w)) {
          val qDestIdx = qReadDestIdxs(w)(cacheVecIdxWidth - 1, 0)
          for (r <- 0 until nRows) {
            aTileBuf(qDestIdx)(r) := readDataWords(w)(precision * (r + 1) - 1, precision * r)
          }
        }
      }
      val nextIdx = qFillIdx + qWordsThisBeat
      when(nextIdx >= dK) {
        qFillIdx := 0.U
        state := s_qk_setup_tile
      }.otherwise {
        qFillIdx := nextIdx
        state := s_req_fill_q
      }
    }
  }

  when(state === s_qk_setup_tile) {
    val remainingCols = kvRows - qkTileBase
    val tileStrideBytes = (maxK * xlenBytes).U(xLen.W)
    activeTileCols := Mux(remainingCols > nCols.U, nCols.U, remainingCols)(colCountWidth - 1, 0)
    bFillBaseAddr := kTilesBase + ((qkTileBase >> tileShift) * tileStrideBytes)
    bFillCount := dK
    bFillIdx := 0.U
    bFillForPV := false.B
    computeKBaseIdx := 0.U
    computeKRemaining := 0.U
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        accumTile(r)(c) := 0.U
      }
    }
    state := s_req_fill_b
  }

  when(state === s_req_fill_b) {
    tlOut.a.valid := true.B
    tlOut.a.bits := bGetBits
    when(tlOut.a.fire) {
      incTLReadReq := true.B
      state := s_wait_fill_b
    }
  }

  when(state === s_wait_fill_b) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp, "fused attention expected AccessAckData on packed-B fill")
      assert(dSource === readTlSourceId.U, "fused attention expected B fill response on read source")
      for (w <- 0 until wordsPerBeat) {
        when(bReadLaneValids(w)) {
          val bDestIdx = bReadDestIdxs(w)(cacheVecIdxWidth - 1, 0)
          for (c <- 0 until nCols) {
            bTileBuf(bDestIdx)(c) := readDataWords(w)(precision * (c + 1) - 1, precision * c)
          }
        }
      }
      val nextIdx = bFillIdx + bWordsThisBeat
      when(nextIdx >= bFillCount) {
        bFillIdx := 0.U
        computeKBaseIdx := 0.U
        computeKRemaining := Mux(bFillForPV, kvRows, dK)
        state := Mux(bFillForPV, s_pv_prep_chunk, s_qk_prep_chunk)
      }.otherwise {
        bFillIdx := nextIdx
        state := s_req_fill_b
      }
    }
  }

  when(state === s_qk_prep_chunk) {
    when(computeKRemaining === 0.U) {
      state := s_qk_store_scores
    }.otherwise {
      computeChunkK := Mux(computeKRemaining > nRows.U, nRows.U, computeKRemaining)(chunkWidth - 1, 0)
      feedRowIdx := 0.U
      captureRowIdx := 0.U
      state := s_qk_load_weights
    }
  }

  when(state === s_qk_load_weights) {
    core.io.loadWeights := true.B
    state := s_qk_feed_rows
  }

  when(state === s_qk_feed_rows) {
    core.io.inValid := true.B
    when(feedRowIdx === (nRows - 1).U) {
      state := s_qk_wait_chunk_out
    }
    feedRowIdx := feedRowIdx + 1.U
  }

  when(state === s_qk_wait_chunk_out) {
    when(captureRowIdx === nRows.U) {
      computeKRemaining := computeKRemaining - computeChunkK
      computeKBaseIdx := computeKBaseIdx + computeChunkK
      state := s_qk_prep_chunk
    }
  }

  when(state === s_qk_store_scores) {
    for (r <- 0 until nRows) {
      when(r.U < qRows) {
        for (c <- 0 until nCols) {
          when(c.U < activeTileCols) {
            val scoreIdx = (qkTileBase + c.U)(cacheVecIdxWidth - 1, 0)
            scoreBuf(r)(scoreIdx) := scorePostBf16(r)(c)
          }
        }
      }
    }
    when(qkTileBase + nCols.U >= kvRows) {
      for (k <- 0 until maxK) {
        for (r <- 0 until nRows) {
          aTileBuf(k)(r) := 0.U
        }
      }
      softRowIdx := 0.U
      softColIdx := 0.U
      softAccum := 0.U
      softMaxPrev := minBf16
      state := s_softmax_row_init
    }.otherwise {
      qkTileBase := qkTileBase + nCols.U
      state := s_qk_setup_tile
    }
  }

  when(state === s_softmax_row_init) {
    softColIdx := 0.U
    softAccum := 0.U
    softMaxPrev := minBf16
    softInvSum := 0.U
    state := s_softmax_p1_load
  }

  when(state === s_softmax_p1_load) {
    val softRow = softRowIdx(rowVecIdxWidth - 1, 0)
    for (lane <- 0 until nCols) {
      val idxFull = softColIdx + lane.U
      val idx = idxFull(cacheVecIdxWidth - 1, 0)
      softLatchedScores(lane) := Mux(idxFull < kvRows, scoreBuf(softRow)(idx), minBf16)
    }
    state := s_softmax_p1_compute
  }

  when(state === s_softmax_p1_compute) {
    val nextCol = softColIdx + nCols.U
    softAccum := softAccumNext
    softMaxPrev := softGlobalMax.io.out
    when(nextCol >= kvRows) {
      softColIdx := 0.U
      softInvSum := softInvSumCandidate
      state := s_softmax_p2_load
    }.otherwise {
      softColIdx := nextCol
      state := s_softmax_p1_load
    }
  }

  when(state === s_softmax_p2_load) {
    val softRow = softRowIdx(rowVecIdxWidth - 1, 0)
    for (lane <- 0 until nCols) {
      val idxFull = softColIdx + lane.U
      val idx = idxFull(cacheVecIdxWidth - 1, 0)
      softLatchedScores(lane) := Mux(idxFull < kvRows, scoreBuf(softRow)(idx), minBf16)
    }
    state := s_softmax_p2_compute
  }

  when(state === s_softmax_p2_compute) {
    val softRow = softRowIdx(rowVecIdxWidth - 1, 0)
    for (lane <- 0 until nCols) {
      val idxFull = softColIdx + lane.U
      val idx = idxFull(cacheVecIdxWidth - 1, 0)
      when(idxFull < kvRows) {
        aTileBuf(idx)(softRow) := softProbBf16(lane)
      }
    }
    val nextCol = softColIdx + nCols.U
    when(nextCol >= kvRows) {
      softColIdx := 0.U
      when(softRowIdx + 1.U >= qRows) {
        state := s_pv_setup
      }.otherwise {
        softRowIdx := softRowIdx + 1.U
        state := s_softmax_row_init
      }
    }.otherwise {
      softColIdx := nextCol
      state := s_softmax_p2_load
    }
  }

  when(state === s_pv_setup) {
    activeTileCols := valueCols
    bFillBaseAddr := vBase
    bFillCount := kvRows
    bFillIdx := 0.U
    bFillForPV := true.B
    computeKBaseIdx := 0.U
    computeKRemaining := 0.U
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        accumTile(r)(c) := 0.U
      }
    }
    state := s_req_fill_b
  }

  when(state === s_pv_prep_chunk) {
    when(computeKRemaining === 0.U) {
      outIdx := 0.U
      state := s_quantize_out
    }.otherwise {
      computeChunkK := Mux(computeKRemaining > nRows.U, nRows.U, computeKRemaining)(chunkWidth - 1, 0)
      feedRowIdx := 0.U
      captureRowIdx := 0.U
      state := s_pv_load_weights
    }
  }

  when(state === s_pv_load_weights) {
    core.io.loadWeights := true.B
    state := s_pv_feed_rows
  }

  when(state === s_pv_feed_rows) {
    core.io.inValid := true.B
    when(feedRowIdx === (nRows - 1).U) {
      state := s_pv_wait_chunk_out
    }
    feedRowIdx := feedRowIdx + 1.U
  }

  when(state === s_pv_wait_chunk_out) {
    when(captureRowIdx === nRows.U) {
      computeKRemaining := computeKRemaining - computeChunkK
      computeKBaseIdx := computeKBaseIdx + computeChunkK
      state := s_pv_prep_chunk
    }
  }

  when(state === s_quantize_out) {
    for (w <- 0 until outputWordCount) {
      packedStoreWordsReg(w) := packedStoreWordsWire(w)
    }
    state := s_req_put
  }

  when(state === s_req_put) {
    tlOut.a.valid := true.B
    tlOut.a.bits := putBits
    when(tlOut.a.fire) {
      incTLWriteReq := true.B
      state := s_wait_put
    }
  }

  when(state === s_wait_put) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsWriteAck, "fused attention expected AccessAck on output write")
      assert(dSource === writeTlSourceId.U, "fused attention expected writeback response on write source")
      val nextOutIdx = outIdx + writeWordsThisBeat
      when(nextOutIdx >= outputWordCount.U) {
        respData := 0.U
        state := s_resp
      }.otherwise {
        outIdx := nextOutIdx
        state := s_req_put
      }
    }
  }

  when(io.resp.fire) {
    state := s_idle
  }

  when(clearPerfCounters) {
    for (i <- 0 until numPerfCounters) {
      perfCounters(i) := 0.U
    }
  }.otherwise {
    when(state =/= s_idle && state =/= s_resp) {
      perfCounters(perfBusyCycles) := perfCounters(perfBusyCycles) + 1.U
    }
    when(incRunCmd) {
      perfCounters(perfRunCmds) := perfCounters(perfRunCmds) + 1.U
    }
    when(incTLReadReq) {
      perfCounters(perfTLReadReqs) := perfCounters(perfTLReadReqs) + 1.U
    }
    when(incTLWriteReq) {
      perfCounters(perfTLWriteReqs) := perfCounters(perfTLWriteReqs) + 1.U
    }
    when(state === s_wait_fill_q) {
      perfCounters(perfWaitQFillCycles) := perfCounters(perfWaitQFillCycles) + 1.U
    }
    when(state === s_wait_fill_b && !bFillForPV) {
      perfCounters(perfWaitKFillCycles) := perfCounters(perfWaitKFillCycles) + 1.U
    }
    when(state === s_qk_load_weights || state === s_qk_feed_rows || state === s_qk_wait_chunk_out) {
      perfCounters(perfQKComputeCycles) := perfCounters(perfQKComputeCycles) + 1.U
    }
    when(state === s_qk_store_scores) {
      perfCounters(perfScorePostCycles) := perfCounters(perfScorePostCycles) + 1.U
    }
    when(state === s_softmax_p1_load || state === s_softmax_p1_compute) {
      perfCounters(perfSoftmaxPass1Cycles) := perfCounters(perfSoftmaxPass1Cycles) + 1.U
    }
    when(state === s_softmax_p2_load || state === s_softmax_p2_compute) {
      perfCounters(perfSoftmaxPass2Cycles) := perfCounters(perfSoftmaxPass2Cycles) + 1.U
    }
    when(state === s_wait_fill_b && bFillForPV) {
      perfCounters(perfWaitVFillCycles) := perfCounters(perfWaitVFillCycles) + 1.U
    }
    when(state === s_pv_load_weights || state === s_pv_feed_rows || state === s_pv_wait_chunk_out) {
      perfCounters(perfPVComputeCycles) := perfCounters(perfPVComputeCycles) + 1.U
    }
    when(state === s_wait_put) {
      perfCounters(perfWaitPutCycles) := perfCounters(perfWaitPutCycles) + 1.U
    }
  }
}

class FusedAttentionWSRoCC(
  opcodes: OpcodeSet,
  val precision: Int = 16,
  val nRows: Int = 4,
  val nCols: Int = 4,
  val maxK: Int = 128,
  val fixedPointFracBits: Int = 8,
  val accumBits: Int = 32,
  val softmaxIntPrecision: Int = 12,
  val softmaxFracPrecision: Int = 20,
  val numTLSourceIds: Int = 2,
  val clientName: String = "FusedAttentionWSRoCC"
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new FusedAttentionWSImpl(this)
  override val atlNode = TLClientNode(
    Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1(
      name = clientName,
      sourceId = IdRange(0, numTLSourceIds)
    ))))
  )
}
