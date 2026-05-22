package girdap

import chisel3._
import chisel3.util._
import scala.math.BigDecimal.RoundingMode

import freechips.rocketchip.diplomacy.IdRange
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import org.chipsalliance.cde.config.Parameters

class FpgaSafeOnlineAttention8x8Impl(
  outer: FpgaSafeOnlineAttention8x8RoCC
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
  private val useSoftmaxExpLut = outer.useSoftmaxExpLut
  private val tlSourceIds = outer.numTLSourceIds
  private val packerEnabled = outer.enablePacker
  private val debugEnabled = outer.enableIntermediateDebug

  private val xlenBytes = xLen / 8
  private val beatBytes = cacheDataBits / 8
  private val elemBytes = precision / 8
  private val beatOffsetBits = log2Ceil(beatBytes)
  private val elemOffsetBits = log2Ceil(elemBytes)
  private val elemsPerBeat = cacheDataBits / precision
  private val wordsPerBeat = beatBytes / xlenBytes
  private val laneWidth = math.max(1, log2Ceil(wordsPerBeat))
  private val tlSourceIdxWidth = math.max(1, log2Ceil(tlSourceIds))
  private val kWidth = log2Ceil(maxK + 1)
  private val rowCountWidth = log2Ceil(nRows + 1)
  private val colCountWidth = log2Ceil(nCols + 1)
  private val tileBaseWidth = log2Ceil(maxK + nCols + 1)
  private val scoreRowIdxWidth = log2Ceil(nRows)
  private val scoreTiles = maxK / nCols
  private val scoreTileIdxWidth = log2Ceil(scoreTiles)
  private val scoreBankAddrWidth = log2Ceil(nRows * scoreTiles)
  private val outCount = nRows * nCols
  private val outputElemsPerWord = xLen / precision
  private val outputWordCount = (outCount + outputElemsPerWord - 1) / outputElemsPerWord
  private val outIdxWidth = log2Ceil(outputWordCount + 1)
  private val scoreFracBits = 2 * fixedPointFracBits
  private val pvProductShift = softmaxFracPrecision + fixedPointFracBits - scoreFracBits
  private val scoreScaleIntBits = 8
  private val softBitWidth = softmaxIntPrecision + softmaxFracPrecision
  private val lutEntries = outer.softmaxRecipLutEntries
  private val lutBits = 64
  private val expLutEntries = outer.softmaxExpLutEntries
  private val expLutRange = outer.softmaxExpLutRange
  private val expLutIndexBits = log2Ceil(expLutEntries)
  private val expLutInputShift = scoreFracBits + log2Ceil(expLutRange) - expLutIndexBits
  private val minScoreFixed = (-(BigInt(1) << (accumPrec - 1))).S(accumPrec.W)
  private val readTlSourceId = 0
  private val writeTlSourceId = 1

  require(precision == 16, "FPGA-safe online attention targets BF16 lanes")
  require(nRows == 8 && nCols == 8, "FPGA-safe online attention targets 8x8 tiles")
  require(xLen == 64, s"software ABI expects RV64 xLen, got $xLen")
  require(cacheDataBits >= precision * nRows, "Q/K/V packed records must fit in one cache beat")
  require(cacheDataBits % xLen == 0, "cacheDataBits must be a multiple of xLen")
  require(isPow2(wordsPerBeat), "wordsPerBeat must be a power of two")
  require(outputElemsPerWord == 4, "BF16 output packing expects four lanes per xLen word")
  require(accumPrec <= xLen, "accumulator must fit in xLen for software-visible counters")
  require(tlSourceIds >= 2, "one read source and one write source are required")
  require(pvProductShift >= 0, "PV product must not need left-shift widening")
  require(isPow2(lutEntries), "softmax reciprocal LUT entries must be a power of two")
  require(isPow2(expLutEntries), "softmax exp LUT entries must be a power of two")
  require(isPow2(expLutRange), "softmax exp LUT range must be a power of two")
  require(expLutInputShift >= 0, "softmax exp LUT index needs more entries than score fixed-point resolution supports")
  require(softmaxFracPrecision >= log2Ceil(lutEntries) - 1, "softmax frac precision too small")
  require(maxK % nCols == 0, "score cache banking expects full-width KV tiles")
  require(isPow2(scoreTiles), "score cache banking expects power-of-two tile count")
  if (packerEnabled) {
    require(cacheDataBits == precision * nRows, "hardware packer expects one 8-lane BF16 record per cache beat")
    require(isPow2(elemBytes), "hardware packer expects power-of-two BF16 element bytes")
  }

  private val qBuf = Reg(Vec(maxK, UInt(cacheDataBits.W)))
  private val kBuf = Reg(Vec(maxK, UInt(cacheDataBits.W)))
  private val vBuf = Reg(Vec(nCols, UInt(cacheDataBits.W)))
  private val scoreAccum = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.S(accumPrec.W))))))
  private val outAccum = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.S(accumPrec.W))))))
  private val rowMax = RegInit(VecInit(Seq.fill(nRows)(minScoreFixed)))
  private val rowDenom = RegInit(VecInit(Seq.fill(nRows)(0.U(softBitWidth.W))))
  // VLSI note: use register-backed score banks while firtool rejects this SyncReadMem path; revert after VLSI runs.
  private val scoreBanks = Reg(Vec(nCols, Vec(nRows * scoreTiles, SInt(accumPrec.W))))
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
  private val packConfigured = RegInit(false.B)
  private val packDimsConfigured = RegInit(false.B)
  private val debugReadProb = RegInit(false.B)

  private val fillIdx = RegInit(0.U(kWidth.W))
  private val computeIdx = RegInit(0.U(kWidth.W))
  private val kvTileBase = RegInit(0.U(tileBaseWidth.W))
  private val activeKvCols = RegInit(0.U(colCountWidth.W))
  private val softRowIdx = RegInit(0.U(rowCountWidth.W))
  private val outIdx = RegInit(0.U(outIdxWidth.W))
  private val respRd = RegInit(0.U(5.W))
  private val respData = RegInit(0.U(xLen.W))

  private val packSrcBase = RegInit(0.U(xLen.W))
  private val packDstBase = RegInit(0.U(xLen.W))
  private val packRows = RegInit(0.U(kWidth.W))
  private val packCols = RegInit(0.U(kWidth.W))
  private val packLd = RegInit(0.U(kWidth.W))
  private val packOutStride = RegInit(0.U(kWidth.W))
  private val packMode = RegInit(0.U(2.W))
  private val packTileBase = RegInit(0.U(tileBaseWidth.W))
  private val packInnerIdx = RegInit(0.U(kWidth.W))
  private val packLane = RegInit(0.U(log2Ceil(nRows).W))
  private val packLaneData = RegInit(VecInit(Seq.fill(nRows)(0.U(precision.W))))
  private val packModeRowMajorTiles = 0.U(2.W)
  private val packModeColumnTiles = 1.U(2.W)
  private val debugLane = RegInit(0.U(log2Ceil(nCols).W))

  private val states = Enum(31)
  private val s_idle = states(0)
  private val s_req_fill_q = states(1)
  private val s_wait_fill_q = states(2)
  private val s_p1_setup_tile = states(3)
  private val s_req_fill_k = states(4)
  private val s_wait_fill_k = states(5)
  private val s_clear_scores = states(6)
  private val s_compute_scores = states(7)
  private val s_p1_row_load = states(8)
  private val s_p1_row_update = states(9)
  private val s_p2_setup_tile = states(10)
  private val s_req_fill_k2 = states(11)
  private val s_wait_fill_k2 = states(12)
  private val s_req_fill_v = states(13)
  private val s_wait_fill_v = states(14)
  private val s_p2_score_latch = states(15)
  private val s_compute_scores2 = states(16)
  private val s_p2_row_load = states(17)
  private val s_p2_row_accum = states(18)
  private val s_quantize = states(19)
  private val s_req_put = states(20)
  private val s_wait_put = states(21)
  private val s_resp = states(22)
  private val s_error = states(23)
  private val s_pack_req_read = states(24)
  private val s_pack_wait_read = states(25)
  private val s_pack_req_put = states(26)
  private val s_pack_wait_put = states(27)
  private val s_dbg_score_load = states(28)
  private val s_dbg_score_latch = states(29)
  private val s_dbg_prob_put = states(30)
  private val state = RegInit(s_idle)

  private val numPerfCounters = 12
  private val perfCounterWidth = log2Ceil(numPerfCounters)
  private val perfCounters = RegInit(VecInit(Seq.fill(numPerfCounters)(0.U(xLen.W))))
  private val clearPerfCounters = WireDefault(false.B)
  private val incTLRead = WireDefault(false.B)
  private val incTLWrite = WireDefault(false.B)
  private val incRun = WireDefault(false.B)

  private val (tlOut, edgesOut) = outer.atlNode.out(0)
  private val dHasData = edgesOut.hasData(tlOut.d.bits)
  private val dSource = tlOut.d.bits.source(tlSourceIdxWidth - 1, 0)
  private val dIsReadResp = dHasData && tlOut.d.bits.opcode === TLMessages.AccessAckData
  private val dIsWriteAck = !dHasData && tlOut.d.bits.opcode === TLMessages.AccessAck

  private def beatAddr(base: UInt, idx: UInt): UInt = base + idx * beatBytes.U
  private def kTileAddr(base: UInt, tileBase: UInt, idx: UInt): UInt =
    base + ((tileBase >> log2Ceil(nCols)) * dK.pad(xLen) + idx.pad(xLen)) * beatBytes.U
  private def wordAddr(base: UInt, idx: UInt): UInt = base + idx * xlenBytes.U
  private def scoreBankAddr(row: UInt, tileBase: UInt): UInt = {
    val tileIdx = tileBase(tileBaseWidth - 1, log2Ceil(nCols))
    Cat(row(scoreRowIdxWidth - 1, 0), tileIdx(scoreTileIdxWidth - 1, 0))(scoreBankAddrWidth - 1, 0)
  }

  private val scoreReadAddr = scoreBankAddr(softRowIdx, kvTileBase)
  private val scoreReadData = Wire(Vec(nCols, SInt(accumPrec.W)))
  for (c <- 0 until nCols) {
    scoreReadData(c) := scoreBanks(c)(scoreReadAddr)
  }

  private val beatLgSize = beatOffsetBits.U
  private val getQBits = edgesOut.Get(readTlSourceId.U, beatAddr(qBase, fillIdx), beatLgSize)._2
  private val getKBits = edgesOut.Get(readTlSourceId.U, kTileAddr(kBase, kvTileBase, fillIdx), beatLgSize)._2
  private val getVBits = edgesOut.Get(readTlSourceId.U, beatAddr(vBase, kvTileBase(kWidth - 1, 0) + fillIdx), beatLgSize)._2

  private val packLaneGlobal = packTileBase + packLane.pad(tileBaseWidth)
  private val packSrcElementIdx = Mux(
    packMode === packModeRowMajorTiles,
    packLaneGlobal.pad(xLen) * packLd.pad(xLen) + packInnerIdx.pad(xLen),
    packInnerIdx.pad(xLen) * packLd.pad(xLen) + packLaneGlobal.pad(xLen)
  )
  private val packSrcByteAddr = packSrcBase + (packSrcElementIdx << elemOffsetBits)
  private val packReadBeatAddr = packSrcByteAddr & ~((beatBytes - 1).U(xLen.W))
  private val packElemLane = packSrcByteAddr(beatOffsetBits - 1, elemOffsetBits)
  private val packReadElems = tlOut.d.bits.data.asTypeOf(Vec(elemsPerBeat, UInt(precision.W)))
  private val packReadElem = packReadElems(packElemLane)
  private val packWriteRecordIdx =
    (packTileBase >> log2Ceil(nRows)).pad(xLen) * packOutStride.pad(xLen) + packInnerIdx.pad(xLen)
  private val packWriteBeatAddr = packDstBase + packWriteRecordIdx * beatBytes.U
  private val packGetBits = edgesOut.Get(readTlSourceId.U, packReadBeatAddr, beatLgSize)._2

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
    putDataWords(w) := Mux(writeLaneValids(w), packedStoreWords(writeLaneWordIdxs(w)), 0.U)
  }
  private val putMaskBytes = Wire(Vec(beatBytes, Bool()))
  for (byte <- 0 until beatBytes) {
    putMaskBytes(byte) := writeLaneValids(byte / xlenBytes)
  }
  private val putBits = edgesOut.Put(
    writeTlSourceId.U,
    writeBeatAddr,
    beatLgSize,
    putDataWords.asUInt,
    putMaskBytes.asUInt
  )._2
  private val packPutBits = edgesOut.Put(
    writeTlSourceId.U,
    packWriteBeatAddr,
    beatLgSize,
    packLaneData.asUInt.pad(cacheDataBits),
    Fill(beatBytes, 1.U(1.W))
  )._2

  private def laneFromBeat(beat: UInt, lane: Int): UInt =
    beat(precision * (lane + 1) - 1, precision * lane)

  private val qLanes = Wire(Vec(nRows, UInt(precision.W)))
  private val kLanes = Wire(Vec(nCols, UInt(precision.W)))
  for (i <- 0 until nRows) {
    qLanes(i) := laneFromBeat(qBuf(computeIdx), i)
    kLanes(i) := laneFromBeat(kBuf(computeIdx), i)
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
      val scaledTrimmed = scaledShifted(accumPrec - 1, 0).asSInt
      scoreFixed(r)(c) := Mux(c.U < activeKvCols, scaledTrimmed, minScoreFixed)
    }
  }

  private val softLatchedScores = Reg(Vec(nCols, SInt(accumPrec.W)))
  private val softVecFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  private val softProbFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  private val rowMaxRead = rowMax(softRowIdx)
  private val rowDenomRead = rowDenom(softRowIdx)

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
    val scaleDown = lutBits - 1 - softmaxFracPrecision
    val lutScaled = if (scaleDown >= 0) (lutVal >> scaleDown) else (lutVal << (-scaleDown))
    val invRaw = lutScaled >> shift
    Mux(nonZero, invRaw.pad(softBitWidth)(softBitWidth - 1, 0), 0.U)
  }

  private val expLut = VecInit(Seq.tabulate(expLutEntries) { i =>
    val x = BigDecimal(i) * BigDecimal(expLutRange) / BigDecimal(math.max(expLutEntries - 1, 1))
    val scaled = (BigDecimal(math.exp(-x.toDouble)) * BigDecimal(2).pow(softmaxFracPrecision))
      .setScale(0, RoundingMode.HALF_UP)
      .toBigInt
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
  private val softMaxFixed = if (useSoftmaxExpLut) {
    expFixedLut(softMaxDiffFixed)
  } else {
    val diffBf16 = Module(new SIntFixedToBFloat16(accumPrec, scoreFracBits))
    val exp = Module(new BFloat16Exp)
    val fixed = Module(new BFloat16ToFixed(softmaxIntPrecision, softmaxFracPrecision))
    diffBf16.io.in := softMaxDiffFixed(accumPrec - 1, 0).asSInt
    exp.io.in := diffBf16.io.out
    fixed.io.in := exp.io.out
    fixed.io.out
  }

  private val softOne = (1.U(softBitWidth.W) << softmaxFracPrecision)(softBitWidth - 1, 0)
  private val softMaxFixedClamped = Mux(softMaxFixed > softOne, softOne, softMaxFixed)
  private val softProdDenomFull = rowDenomRead * softMaxFixedClamped
  private val softProdDenom = (softProdDenomFull >> softmaxFracPrecision)(softBitWidth - 1, 0)
  private val softSubBase = Mux(state === s_p2_row_accum || state === s_dbg_prob_put, rowMaxRead, softGlobalMaxFixed)
  private val softInvSum = reciprocalFixed(rowDenomRead)

  for (i <- 0 until nCols) {
    val laneValid = i.U < activeKvCols
    val diff = softLatchedScores(i) - softSubBase
    val expValue = Wire(UInt(softBitWidth.W))
    if (useSoftmaxExpLut) {
      expValue := expFixedLut(diff)
    } else {
      val diffBf16 = Module(new SIntFixedToBFloat16(accumPrec, scoreFracBits))
      val exp = Module(new BFloat16Exp)
      val fixed = Module(new BFloat16ToFixed(softmaxIntPrecision, softmaxFracPrecision))
      diffBf16.io.in := diff(accumPrec - 1, 0).asSInt
      exp.io.in := diffBf16.io.out
      fixed.io.in := exp.io.out
      expValue := fixed.io.out
    }
    softVecFixed(i) := Mux(laneValid, expValue, 0.U)
    val normFull = softVecFixed(i) * softInvSum
    val normFixed = (normFull >> softmaxFracPrecision)(softBitWidth - 1, 0)
    softProbFixed(i) := Mux(laneValid, normFixed, 0.U)
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
      if (idx < outCount) {
        packedElems(e) := outBf16(idx)
      } else {
        packedElems(e) := 0.U
      }
    }
    packedStoreWordsWire(w) := packedElems.asUInt
  }

  tlOut.a.valid := false.B
  tlOut.a.bits := getQBits
  tlOut.d.ready := false.B
  io.cmd.ready := state === s_idle
  io.resp.valid := state === s_resp
  io.resp.bits.rd := respRd
  io.resp.bits.data := respData
  io.busy := state =/= s_idle
  io.interrupt := false.B

  when(io.cmd.fire) {
    respRd := io.cmd.bits.inst.rd
    val funct = io.cmd.bits.inst.funct
    when(funct === 8.U) {
      qBase := io.cmd.bits.rs1
      kBase := io.cmd.bits.rs2
      qkConfigured := true.B
      scoresReady := false.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 9.U) {
      vBase := io.cmd.bits.rs1
      outBase := io.cmd.bits.rs2
      voutConfigured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 10.U) {
      val newQRows = io.cmd.bits.rs1(15, 0)(rowCountWidth - 1, 0)
      val newKvRows = io.cmd.bits.rs1(31, 16)(kWidth - 1, 0)
      val newDK = io.cmd.bits.rs1(47, 32)(kWidth - 1, 0)
      val newValueCols = io.cmd.bits.rs1(63, 48)(colCountWidth - 1, 0)
      when(newQRows =/= qRows || newKvRows =/= kvRows || newDK =/= dK) {
        scoresReady := false.B
      }
      qRows := newQRows
      kvRows := newKvRows
      dK := newDK
      valueCols := newValueCols
      dimsConfigured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 11.U) {
      scaleBf16 := io.cmd.bits.rs1(precision - 1, 0)
      scoresReady := false.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 12.U) {
      incRun := true.B
      val dimsValid =
        qkConfigured && voutConfigured && dimsConfigured &&
          (qRows =/= 0.U) && (qRows <= nRows.U) &&
          (kvRows =/= 0.U) && (kvRows <= maxK.U) &&
          (dK =/= 0.U) && (dK <= maxK.U) &&
          (valueCols =/= 0.U) && (valueCols <= nCols.U)
      respData := Mux(!qkConfigured || !voutConfigured || !dimsConfigured, 1.U, Mux(!dimsValid, 2.U, 0.U))
      when(dimsValid) {
        scoresReady := false.B
        applyAfterScores := true.B
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
        state := s_req_fill_q
      }.otherwise {
        state := s_resp
      }
    }.elsewhen(funct === 13.U) {
      incRun := true.B
      val dimsValid =
        qkConfigured && dimsConfigured &&
          (qRows =/= 0.U) && (qRows <= nRows.U) &&
          (kvRows =/= 0.U) && (kvRows <= maxK.U) &&
          (dK =/= 0.U) && (dK <= maxK.U)
      respData := Mux(!qkConfigured || !dimsConfigured, 1.U, Mux(!dimsValid, 2.U, 0.U))
      when(dimsValid) {
        scoresReady := false.B
        applyAfterScores := false.B
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
        state := s_req_fill_q
      }.otherwise {
        state := s_resp
      }
    }.elsewhen(funct === 14.U) {
      incRun := true.B
      val dimsValid =
        voutConfigured && dimsConfigured && scoresReady &&
          (qRows =/= 0.U) && (qRows <= nRows.U) &&
          (kvRows =/= 0.U) && (kvRows <= maxK.U) &&
          (dK =/= 0.U) && (dK <= maxK.U) &&
          (valueCols =/= 0.U) && (valueCols <= nCols.U)
      val missingConfig = !voutConfigured || !dimsConfigured
      respData := Mux(missingConfig, 1.U, Mux(!scoresReady, 3.U, Mux(!dimsValid, 2.U, 0.U)))
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
        state := s_p2_setup_tile
      }.otherwise {
        state := s_resp
      }
    }.elsewhen(if (packerEnabled) funct === 15.U else false.B) {
      packSrcBase := io.cmd.bits.rs1
      packDstBase := io.cmd.bits.rs2
      packConfigured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(if (packerEnabled) funct === 16.U else false.B) {
      packRows := io.cmd.bits.rs1(15, 0)(kWidth - 1, 0)
      packCols := io.cmd.bits.rs1(31, 16)(kWidth - 1, 0)
      packLd := io.cmd.bits.rs1(47, 32)(kWidth - 1, 0)
      packMode := io.cmd.bits.rs1(49, 48)
      packOutStride := io.cmd.bits.rs2(15, 0)(kWidth - 1, 0)
      packDimsConfigured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(if (packerEnabled) funct === 17.U else false.B) {
      incRun := true.B
      val modeValid = packMode === packModeRowMajorTiles || packMode === packModeColumnTiles
      val ldValid = Mux(packMode === packModeRowMajorTiles, packLd >= packCols, packLd >= packRows)
      val dimsValid =
        packConfigured && packDimsConfigured && modeValid && ldValid &&
          (packRows =/= 0.U) && (packRows <= maxK.U) &&
          (packCols =/= 0.U) && (packCols <= maxK.U) &&
          (packOutStride >= packCols) && (packOutStride <= maxK.U)
      respData := Mux(!packConfigured || !packDimsConfigured, 1.U, Mux(!dimsValid, 2.U, 0.U))
      when(dimsValid) {
        packTileBase := 0.U
        packInnerIdx := 0.U
        packLane := 0.U
        for (i <- 0 until nRows) {
          packLaneData(i) := 0.U
        }
        state := s_pack_req_read
      }.otherwise {
        state := s_resp
      }
    }.elsewhen(if (debugEnabled) (funct === 18.U || funct === 19.U) else false.B) {
      incRun := true.B
      val newDebugRow = io.cmd.bits.rs1(15, 0)(rowCountWidth - 1, 0)
      val newDebugCol = io.cmd.bits.rs1(31, 16)(kWidth - 1, 0)
      val dimsValid =
        dimsConfigured && scoresReady &&
          (qRows =/= 0.U) && (qRows <= nRows.U) &&
          (kvRows =/= 0.U) && (kvRows <= maxK.U) &&
          (dK =/= 0.U) && (dK <= maxK.U) &&
          (newDebugRow < qRows) && (newDebugCol < kvRows)
      respData := Mux(!dimsConfigured, 1.U, Mux(!scoresReady, 3.U, Mux(!dimsValid, 2.U, 0.U)))
      when(dimsValid) {
        val colTileBase = (newDebugCol >> log2Ceil(nCols)) << log2Ceil(nCols)
        kvTileBase := colTileBase(tileBaseWidth - 1, 0)
        activeKvCols := nCols.U
        softRowIdx := newDebugRow
        debugLane := newDebugCol(log2Ceil(nCols) - 1, 0)
        debugReadProb := funct === 19.U
        state := s_dbg_score_load
      }.otherwise {
        state := s_resp
      }
    }.elsewhen(funct === 5.U) {
      clearPerfCounters := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 6.U) {
      val counterSel = io.cmd.bits.rs1
      val counterIdx = counterSel(perfCounterWidth - 1, 0)
      respData := Mux(counterSel < numPerfCounters.U, perfCounters(counterIdx), "hBAD0BAD0BAD0BAD0".U)
      state := s_resp
    }.elsewhen(funct === 2.U) {
      respData := Cat(0.U((xLen - 2).W), state =/= s_idle, state === s_resp)
      state := s_resp
    }.otherwise {
      respData := "hDEAD".U
      state := s_resp
    }
  }

  when(state === s_pack_req_read) {
    val laneValid = packLaneGlobal < packRows.pad(tileBaseWidth)
    when(!laneValid) {
      packLaneData(packLane) := 0.U
      when(packLane === (nRows - 1).U) {
        packLane := 0.U
        state := s_pack_req_put
      }.otherwise {
        packLane := packLane + 1.U
      }
    }.otherwise {
      tlOut.a.valid := true.B
      tlOut.a.bits := packGetBits
      when(tlOut.a.fire) {
        incTLRead := true.B
        state := s_pack_wait_read
      }
    }
  }

  when(state === s_pack_wait_read) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp && dSource === readTlSourceId.U, "online attention packer expected read response")
      packLaneData(packLane) := packReadElem
      when(packLane === (nRows - 1).U) {
        packLane := 0.U
        state := s_pack_req_put
      }.otherwise {
        packLane := packLane + 1.U
        state := s_pack_req_read
      }
    }
  }

  when(state === s_pack_req_put) {
    tlOut.a.valid := true.B
    tlOut.a.bits := packPutBits
    when(tlOut.a.fire) {
      incTLWrite := true.B
      state := s_pack_wait_put
    }
  }

  when(state === s_pack_wait_put) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsWriteAck && dSource === writeTlSourceId.U, "online attention packer expected write ack")
      val nextInnerIdx = packInnerIdx + 1.U
      when(nextInnerIdx >= packCols) {
        val nextTileBase = packTileBase + nRows.U
        packInnerIdx := 0.U
        when(nextTileBase >= packRows.pad(tileBaseWidth)) {
          respData := 0.U
          state := s_resp
        }.otherwise {
          packTileBase := nextTileBase
          packLane := 0.U
          state := s_pack_req_read
        }
      }.otherwise {
        packInnerIdx := nextInnerIdx
        packLane := 0.U
        state := s_pack_req_read
      }
    }
  }

  when(state === s_dbg_score_load) {
    state := s_dbg_score_latch
  }

  when(state === s_dbg_score_latch) {
    for (c <- 0 until nCols) {
      softLatchedScores(c) := Mux(c.U < activeKvCols, scoreReadData(c), minScoreFixed)
    }
    state := s_dbg_prob_put
  }

  when(state === s_dbg_prob_put) {
    respData := Mux(debugReadProb, softProbFixed(debugLane), softLatchedScores(debugLane).asUInt)
    state := s_resp
  }

  when(state === s_req_fill_q) {
    tlOut.a.valid := fillIdx < dK
    tlOut.a.bits := getQBits
    when(tlOut.a.fire) {
      incTLRead := true.B
      state := s_wait_fill_q
    }
  }

  when(state === s_wait_fill_q) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp && dSource === readTlSourceId.U, "online attention expected Q read response")
      qBuf(fillIdx) := tlOut.d.bits.data
      val nextIdx = fillIdx + 1.U
      when(nextIdx >= dK) {
        fillIdx := 0.U
        kvTileBase := 0.U
        state := s_p1_setup_tile
      }.otherwise {
        fillIdx := nextIdx
        state := s_req_fill_q
      }
    }
  }

  when(state === s_p1_setup_tile || state === s_p2_setup_tile) {
    val remaining = kvRows.pad(tileBaseWidth) - kvTileBase
    activeKvCols := Mux(remaining > nCols.U, nCols.U, remaining)(colCountWidth - 1, 0)
    fillIdx := 0.U
    when(state === s_p2_setup_tile) {
      softRowIdx := 0.U
    }
    state := Mux(state === s_p1_setup_tile, s_req_fill_k, s_req_fill_v)
  }

  when(state === s_req_fill_k || state === s_req_fill_k2) {
    tlOut.a.valid := fillIdx < dK
    tlOut.a.bits := getKBits
    when(tlOut.a.fire) {
      incTLRead := true.B
      state := Mux(state === s_req_fill_k, s_wait_fill_k, s_wait_fill_k2)
    }
  }

  when(state === s_wait_fill_k || state === s_wait_fill_k2) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp && dSource === readTlSourceId.U, "online attention expected K read response")
      kBuf(fillIdx) := tlOut.d.bits.data
      val nextIdx = fillIdx + 1.U
      when(nextIdx >= dK) {
        fillIdx := 0.U
        computeIdx := 0.U
        state := Mux(state === s_wait_fill_k, s_clear_scores, s_req_fill_v)
      }.otherwise {
        fillIdx := nextIdx
        state := Mux(state === s_wait_fill_k, s_req_fill_k, s_req_fill_k2)
      }
    }
  }

  when(state === s_req_fill_v) {
    tlOut.a.valid := fillIdx < activeKvCols
    tlOut.a.bits := getVBits
    when(tlOut.a.fire) {
      incTLRead := true.B
      state := s_wait_fill_v
    }
  }

  when(state === s_wait_fill_v) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp && dSource === readTlSourceId.U, "online attention expected V read response")
      vBuf(fillIdx) := tlOut.d.bits.data
      val nextIdx = fillIdx + 1.U
      when(nextIdx >= activeKvCols) {
        fillIdx := 0.U
        computeIdx := 0.U
        state := s_p2_row_load
      }.otherwise {
        fillIdx := nextIdx
        state := s_req_fill_v
      }
    }
  }

  when(state === s_clear_scores) {
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        scoreAccum(r)(c) := 0.S
      }
    }
    computeIdx := 0.U
    state := s_compute_scores
  }

  when(state === s_compute_scores) {
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        val product = qFixed(r).io.out * kFixed(c).io.out
        val sum = scoreAccum(r)(c) + product.asSInt
        scoreAccum(r)(c) := sum(accumPrec - 1, 0).asSInt
      }
    }
    when(computeIdx + 1.U >= dK) {
      softRowIdx := 0.U
      state := s_p1_row_load
    }.otherwise {
      computeIdx := computeIdx + 1.U
    }
  }

  when(state === s_p1_row_load) {
    for (c <- 0 until nCols) {
      softLatchedScores(c) := scoreFixed(softRowIdx)(c)
    }
    state := s_p1_row_update
  }

  when(state === s_p2_row_load) {
    state := s_p2_score_latch
  }

  when(state === s_p2_score_latch) {
    for (c <- 0 until nCols) {
      softLatchedScores(c) := Mux(c.U < activeKvCols, scoreReadData(c), minScoreFixed)
    }
    state := s_p2_row_accum
  }

  when(state === s_p1_row_update) {
    when(softRowIdx < qRows) {
      val writeAddr = scoreBankAddr(softRowIdx, kvTileBase)
      for (c <- 0 until nCols) {
        val scoreIdx = kvTileBase + c.U
        when(c.U < activeKvCols && scoreIdx < kvRows) {
          scoreBanks(c)(writeAddr) := softLatchedScores(c)
        }
      }
      rowMax(softRowIdx) := softGlobalMaxFixed
      rowDenom(softRowIdx) := softDenomNext
    }
    when(softRowIdx + 1.U >= qRows) {
      val nextTile = kvTileBase + nCols.U
      when(nextTile >= kvRows) {
        kvTileBase := 0.U
        scoresReady := true.B
        when(applyAfterScores) {
          state := s_p2_setup_tile
        }.otherwise {
          respData := 0.U
          state := s_resp
        }
      }.otherwise {
        kvTileBase := nextTile
        state := s_p1_setup_tile
      }
    }.otherwise {
      softRowIdx := softRowIdx + 1.U
      state := s_p1_row_load
    }
  }

  when(state === s_p2_row_accum) {
    when(softRowIdx < qRows) {
      for (vc <- 0 until nCols) {
        val terms = Seq.tabulate(nCols) { kv =>
          val product = (softProbFixed(kv).asSInt * vFixed(kv)(vc).io.out).asSInt
          val shiftedProduct =
            if (pvProductShift == 0) product else (product >> pvProductShift).asSInt
          val addend = Wire(SInt(accumPrec.W))
          addend := Mux(
            kv.U < activeKvCols && vc.U < valueCols,
            shiftedProduct.pad(accumPrec).asSInt,
            0.S(accumPrec.W)
          )
          addend
        }
        val tileSum = terms.reduce(_ + _)
        val sum = outAccum(softRowIdx)(vc) + tileSum
        outAccum(softRowIdx)(vc) := sum(accumPrec - 1, 0).asSInt
      }
    }
    when(softRowIdx + 1.U >= qRows) {
      val nextTile = kvTileBase + nCols.U
      when(nextTile >= kvRows) {
        outIdx := 0.U
        state := s_quantize
      }.otherwise {
        kvTileBase := nextTile
        state := s_p2_setup_tile
      }
    }.otherwise {
      softRowIdx := softRowIdx + 1.U
      state := s_p2_row_load
    }
  }

  when(state === s_quantize) {
    for (w <- 0 until outputWordCount) {
      packedStoreWords(w) := packedStoreWordsWire(w)
    }
    outIdx := 0.U
    state := s_req_put
  }

  when(state === s_req_put) {
    tlOut.a.valid := outIdx < outputWordCount.U
    tlOut.a.bits := putBits
    when(tlOut.a.fire) {
      incTLWrite := true.B
      state := s_wait_put
    }
  }

  when(state === s_wait_put) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsWriteAck && dSource === writeTlSourceId.U, "online attention expected output write ack")
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
    when(state =/= s_idle && state =/= s_resp) { perfCounters(0) := perfCounters(0) + 1.U }
    when(incRun) { perfCounters(1) := perfCounters(1) + 1.U }
    when(incTLRead) { perfCounters(2) := perfCounters(2) + 1.U }
    when(incTLWrite) { perfCounters(3) := perfCounters(3) + 1.U }
    when(state === s_compute_scores) { perfCounters(4) := perfCounters(4) + 1.U }
    when(state === s_p1_row_update) { perfCounters(5) := perfCounters(5) + 1.U }
    when(state === s_p2_row_accum) { perfCounters(6) := perfCounters(6) + 1.U }
    when(state === s_wait_fill_q) { perfCounters(7) := perfCounters(7) + 1.U }
    when(state === s_wait_fill_k || state === s_wait_fill_k2) { perfCounters(8) := perfCounters(8) + 1.U }
    when(state === s_wait_fill_v) { perfCounters(9) := perfCounters(9) + 1.U }
    when(state === s_wait_put) { perfCounters(10) := perfCounters(10) + 1.U }
  }
}

class FpgaSafeOnlineAttention8x8RoCC(
  opcodes: OpcodeSet,
  val precision: Int = 16,
  val nRows: Int = 8,
  val nCols: Int = 8,
  val maxK: Int = 256,
  val fixedPointFracBits: Int = 8,
  val accumBits: Int = 64,
  val softmaxIntPrecision: Int = 32,
  val softmaxFracPrecision: Int = 32,
  val softmaxRecipLutEntries: Int = 256,
  val useSoftmaxExpLut: Boolean = false,
  val softmaxExpLutEntries: Int = 1024,
  val softmaxExpLutRange: Int = 16,
  val numTLSourceIds: Int = 2,
  val enablePacker: Boolean = false,
  val enableIntermediateDebug: Boolean = false,
  val clientName: String = "FpgaSafeOnlineAttention8x8RoCC"
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new FpgaSafeOnlineAttention8x8Impl(this)
  override val atlNode = TLClientNode(
    Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1(
      name = clientName,
      sourceId = IdRange(0, numTLSourceIds)
    ))))
  )
}
