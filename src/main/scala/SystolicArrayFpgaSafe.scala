package girdap

import chisel3._
import chisel3.util._
import scala.math.BigDecimal.RoundingMode

import freechips.rocketchip.diplomacy.IdRange
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink.{TLClientNode, TLMasterParameters, TLMasterPortParameters, TLMessages}
import org.chipsalliance.cde.config.Parameters

class SystolicArrayFpgaSafeImpl(
  outer: SystolicArrayFpgaSafeRoCC
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
  private val tlSourceIds = outer.numTLSourceIds

  private val kWidth = log2Ceil(maxK + 1)
  private val outCount = nRows * nCols
  private val outputElemsPerWord = xLen / precision
  private val outputWordCount = (outCount + outputElemsPerWord - 1) / outputElemsPerWord
  private val outIdxWidth = log2Ceil(outputWordCount + 1)
  private val xlenBytes = xLen / 8
  private val lgXlenBytes = log2Ceil(xlenBytes)
  private val beatBytes = cacheDataBits / 8
  private val wordsPerBeat = beatBytes / xlenBytes
  private val laneWidth = math.max(1, log2Ceil(wordsPerBeat))
  private val beatOffsetBits = log2Ceil(beatBytes)
  private val tlSourceIdxWidth = math.max(1, log2Ceil(tlSourceIds))
  private val readTlSourceId = 0
  private val writeTlSourceId = 1

  require(precision == 16, "FPGA-safe matmul currently targets BF16 lanes")
  require(nRows == 4 && nCols == 4, "FPGA-safe matmul currently targets one 4x4 tile")
  require(maxK > 0, "maxK must be positive")
  require(accumPrec <= xLen, s"accumBits must fit in xLen (${accumPrec} > ${xLen})")
  require(outputElemsPerWord > 0, "xLen must hold at least one BF16 output")
  require(cacheDataBits % xLen == 0, s"cacheDataBits must be a multiple of xLen (${cacheDataBits} % ${xLen})")
  require(isPow2(wordsPerBeat), s"wordsPerBeat must be a power of two, got ${wordsPerBeat}")
  require(tlSourceIds >= 2, "FPGA-safe matmul uses one read source and one write source")

  val aBuf = Reg(Vec(maxK, UInt(xLen.W)))
  val bBuf = Reg(Vec(maxK, UInt(xLen.W)))
  val accum = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.S(accumPrec.W))))))
  val packedStoreWords = Reg(Vec(outputWordCount, UInt(xLen.W)))

  val aBase = RegInit(0.U(xLen.W))
  val bBase = RegInit(0.U(xLen.W))
  val cBase = RegInit(0.U(xLen.W))
  val totalK = RegInit(0.U(kWidth.W))
  val bCacheK = RegInit(0.U(kWidth.W))
  val bCacheValid = RegInit(false.B)
  val configured = RegInit(false.B)
  val bFillForRun = RegInit(false.B)

  val fillIdx = RegInit(0.U(kWidth.W))
  val computeIdx = RegInit(0.U(kWidth.W))
  val outIdx = RegInit(0.U(outIdxWidth.W))
  val respRd = RegInit(0.U(5.W))
  val respData = RegInit(0.U(xLen.W))

  val (s_idle :: s_req_fill_b :: s_wait_fill_b :: s_req_fill_a :: s_wait_fill_a ::
    s_compute :: s_quantize :: s_req_put :: s_wait_put :: s_resp :: Nil) = Enum(10)
  val state = RegInit(s_idle)

  // Keep this mapping aligned with software/include/systolic_ws.h.
  private val numPerfCounters = 15
  private val perfCounterWidth = log2Ceil(numPerfCounters)
  private val perfBusyCycles = 0
  private val perfRunCmds = 1
  private val perfPreloadCmds = 2
  private val perfPreloadReuseHits = 3
  private val perfChunksStarted = 4
  private val perfFeedCycles = 5
  private val perfCaptureRows = 6
  private val perfTLBReads = 7
  private val perfTLAReads = 8
  private val perfTLCWrites = 9
  private val perfWaitFillBCycles = 10
  private val perfWaitFillACycles = 11
  private val perfLoadWeightCycles = 12
  private val perfWaitChunkOutCycles = 13
  private val perfWaitPutCycles = 14
  val perfCounters = RegInit(VecInit(Seq.fill(numPerfCounters)(0.U(xLen.W))))

  val clearPerfCounters = WireDefault(false.B)
  val incRunCmd = WireDefault(false.B)
  val incPreloadCmd = WireDefault(false.B)
  val incPreloadReuseHit = WireDefault(false.B)
  val incChunkStarted = WireDefault(false.B)
  val incTLBRead = WireDefault(false.B)
  val incTLARead = WireDefault(false.B)
  val incTLCWrite = WireDefault(false.B)

  val (tlOut, edgesOut) = outer.atlNode.out(0)
  val dHasData = edgesOut.hasData(tlOut.d.bits)
  val dSource = tlOut.d.bits.source(tlSourceIdxWidth - 1, 0)
  val dIsReadResp = dHasData && tlOut.d.bits.opcode === TLMessages.AccessAckData
  val dIsWriteAck = !dHasData && tlOut.d.bits.opcode === TLMessages.AccessAck

  val readDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    readDataWords(w) := tlOut.d.bits.data((w + 1) * xLen - 1, w * xLen)
  }

  def beatBaseIdx(idx: UInt): UInt = {
    if (wordsPerBeat > 1) {
      (idx >> laneWidth) << laneWidth
    } else {
      idx
    }
  }

  def wordAddr(base: UInt, idx: UInt): UInt = base + idx * xlenBytes.U

  val readLgSize = if (wordsPerBeat > 1) beatOffsetBits.U else lgXlenBytes.U

  val bReadAddr = wordAddr(bBase, fillIdx)
  val bReadLane = if (wordsPerBeat > 1) bReadAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  val bReadBeatAddr = wordAddr(bBase, beatBaseIdx(fillIdx))
  val getBBits = edgesOut.Get(
    fromSource = readTlSourceId.U(tlSourceIdxWidth.W),
    toAddress = bReadBeatAddr,
    lgSize = readLgSize
  )._2

  val aReadAddr = wordAddr(aBase, fillIdx)
  val aReadLane = if (wordsPerBeat > 1) aReadAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  val aReadBeatAddr = wordAddr(aBase, beatBaseIdx(fillIdx))
  val getABits = edgesOut.Get(
    fromSource = readTlSourceId.U(tlSourceIdxWidth.W),
    toAddress = aReadBeatAddr,
    lgSize = readLgSize
  )._2

  val bReadDestIdxs = Wire(Vec(wordsPerBeat, UInt(kWidth.W)))
  val bReadLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  val aReadDestIdxs = Wire(Vec(wordsPerBeat, UInt(kWidth.W)))
  val aReadLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    if (wordsPerBeat > 1) {
      val bLaneAfterStart = w.U(laneWidth.W) >= bReadLane
      val bLaneDelta = Mux(bLaneAfterStart, w.U(laneWidth.W) - bReadLane, 0.U(laneWidth.W))
      bReadDestIdxs(w) := fillIdx + bLaneDelta
      bReadLaneValids(w) := bLaneAfterStart && bReadDestIdxs(w) < bCacheK

      val aLaneAfterStart = w.U(laneWidth.W) >= aReadLane
      val aLaneDelta = Mux(aLaneAfterStart, w.U(laneWidth.W) - aReadLane, 0.U(laneWidth.W))
      aReadDestIdxs(w) := fillIdx + aLaneDelta
      aReadLaneValids(w) := aLaneAfterStart && aReadDestIdxs(w) < totalK
    } else {
      bReadDestIdxs(w) := fillIdx
      bReadLaneValids(w) := fillIdx < bCacheK
      aReadDestIdxs(w) := fillIdx
      aReadLaneValids(w) := fillIdx < totalK
    }
  }
  val bWordsThisBeat = PopCount(bReadLaneValids)
  val aWordsThisBeat = PopCount(aReadLaneValids)

  val writeAddr = wordAddr(cBase, outIdx)
  val writeLane = if (wordsPerBeat > 1) writeAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  val writeBeatAddr = if (wordsPerBeat > 1) {
    writeAddr & ~((beatBytes - 1).U(xLen.W))
  } else {
    writeAddr
  }
  val writeLaneWordIdxs = Wire(Vec(wordsPerBeat, UInt(outIdxWidth.W)))
  val writeLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    if (wordsPerBeat > 1) {
      val laneAfterStart = w.U(laneWidth.W) >= writeLane
      val laneDelta = Mux(laneAfterStart, w.U(laneWidth.W) - writeLane, 0.U(laneWidth.W))
      writeLaneWordIdxs(w) := outIdx + laneDelta
      writeLaneValids(w) := laneAfterStart && writeLaneWordIdxs(w) < outputWordCount.U
    } else {
      writeLaneWordIdxs(w) := outIdx
      writeLaneValids(w) := outIdx < outputWordCount.U
    }
  }
  val writeWordsThisBeat = PopCount(writeLaneValids)
  val putDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    putDataWords(w) := Mux(writeLaneValids(w), packedStoreWords(writeLaneWordIdxs(w)), 0.U)
  }
  val putMaskBytes = Wire(Vec(beatBytes, Bool()))
  for (byte <- 0 until beatBytes) {
    putMaskBytes(byte) := writeLaneValids(byte / xlenBytes)
  }
  val putBits = edgesOut.Put(
    fromSource = writeTlSourceId.U(tlSourceIdxWidth.W),
    toAddress = writeBeatAddr,
    lgSize = readLgSize,
    data = putDataWords.asUInt,
    mask = putMaskBytes.asUInt
  )._2

  val aLanes = Wire(Vec(nRows, UInt(precision.W)))
  val bLanes = Wire(Vec(nCols, UInt(precision.W)))
  for (r <- 0 until nRows) {
    aLanes(r) := aBuf(computeIdx)(precision * (r + 1) - 1, precision * r)
  }
  for (c <- 0 until nCols) {
    bLanes(c) := bBuf(computeIdx)(precision * (c + 1) - 1, precision * c)
  }

  val aFixed = Seq.fill(nRows)(Module(new BFloat16ToSIntFixed(intBits = precision - fixedPointFracBits, fracBits = fixedPointFracBits)))
  val bFixed = Seq.fill(nCols)(Module(new BFloat16ToSIntFixed(intBits = precision - fixedPointFracBits, fracBits = fixedPointFracBits)))
  for (r <- 0 until nRows) {
    aFixed(r).io.in := aLanes(r)
  }
  for (c <- 0 until nCols) {
    bFixed(c).io.in := bLanes(c)
  }

  val bf16Out = Wire(Vec(outCount, UInt(precision.W)))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      val conv = Module(new SIntFixedToBFloat16(accumPrec, 2 * fixedPointFracBits))
      conv.io.in := accum(r)(c)
      bf16Out(r * nCols + c) := conv.io.out
    }
  }
  val packedStoreWordsWire = Wire(Vec(outputWordCount, UInt(xLen.W)))
  for (w <- 0 until outputWordCount) {
    val packedElems = Wire(Vec(outputElemsPerWord, UInt(precision.W)))
    for (e <- 0 until outputElemsPerWord) {
      val idx = w * outputElemsPerWord + e
      if (idx < outCount) {
        packedElems(e) := bf16Out(idx)
      } else {
        packedElems(e) := 0.U
      }
    }
    packedStoreWordsWire(w) := packedElems.asUInt
  }

  tlOut.a.valid := false.B
  tlOut.a.bits := getABits
  tlOut.d.ready := false.B

  io.cmd.ready := state === s_idle
  io.resp.valid := state === s_resp
  io.resp.bits.rd := respRd
  io.resp.bits.data := respData

  when(io.cmd.fire) {
    respRd := io.cmd.bits.inst.rd
    val funct = io.cmd.bits.inst.funct

    when(funct === 0.U) {
      aBase := io.cmd.bits.rs1
      bBase := io.cmd.bits.rs2
      configured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 1.U) {
      incRunCmd := true.B
      val requestedK = io.cmd.bits.rs2
      cBase := io.cmd.bits.rs1
      totalK := requestedK(kWidth - 1, 0)
      bCacheK := requestedK(kWidth - 1, 0)
      fillIdx := 0.U
      computeIdx := 0.U
      outIdx := 0.U
      bFillForRun := true.B
      bCacheValid := false.B
      for (r <- 0 until nRows) {
        for (c <- 0 until nCols) {
          accum(r)(c) := 0.S
        }
      }
      respData := Mux(!configured, 1.U, Mux(requestedK > maxK.U, 2.U, 0.U))
      state := Mux(!configured || requestedK > maxK.U, s_resp, Mux(requestedK === 0.U, s_quantize, s_req_fill_b))
    }.elsewhen(funct === 3.U) {
      incPreloadCmd := true.B
      val requestedK = io.cmd.bits.rs2
      bBase := io.cmd.bits.rs1
      bCacheK := requestedK(kWidth - 1, 0)
      fillIdx := 0.U
      bFillForRun := false.B
      bCacheValid := requestedK === 0.U
      configured := true.B
      respData := Mux(requestedK > maxK.U, 2.U, 0.U)
      state := Mux(requestedK > maxK.U || requestedK === 0.U, s_resp, s_req_fill_b)
    }.elsewhen(funct === 6.U) {
      incRunCmd := true.B
      incPreloadReuseHit := bCacheValid
      aBase := io.cmd.bits.rs1
      cBase := io.cmd.bits.rs2
      totalK := bCacheK
      fillIdx := 0.U
      computeIdx := 0.U
      outIdx := 0.U
      for (r <- 0 until nRows) {
        for (c <- 0 until nCols) {
          accum(r)(c) := 0.S
        }
      }
      respData := Mux(!bCacheValid, 3.U, 0.U)
      state := Mux(!bCacheValid, s_resp, Mux(bCacheK === 0.U, s_quantize, s_req_fill_a))
    }.elsewhen(funct === 2.U) {
      respData := Cat(0.U((xLen - 2).W), state =/= s_idle, state === s_resp)
      state := s_resp
    }.elsewhen(funct === 4.U) {
      val counterSel = io.cmd.bits.rs1
      val counterIdx = counterSel(perfCounterWidth - 1, 0)
      respData := Mux(
        counterSel < numPerfCounters.U,
        perfCounters(counterIdx),
        "hBAD0BAD0BAD0BAD0".U(xLen.W)
      )
      state := s_resp
    }.elsewhen(funct === 5.U) {
      clearPerfCounters := true.B
      respData := 0.U
      state := s_resp
    }.otherwise {
      respData := "hDEAD".U
      state := s_resp
    }
  }

  when(state === s_req_fill_b) {
    tlOut.a.valid := fillIdx < bCacheK
    tlOut.a.bits := getBBits
    when(tlOut.a.fire) {
      incTLBRead := true.B
      state := s_wait_fill_b
    }
  }

  when(state === s_wait_fill_b) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp, "FPGA-safe matmul expected AccessAckData on B-fill")
      assert(dSource === readTlSourceId.U(tlSourceIdxWidth.W), "FPGA-safe matmul received unexpected B-fill source")
      for (w <- 0 until wordsPerBeat) {
        when(bReadLaneValids(w)) {
          bBuf(bReadDestIdxs(w)) := readDataWords(w)
        }
      }
      val nextIdx = fillIdx + bWordsThisBeat
      when(nextIdx >= bCacheK) {
        fillIdx := 0.U
        bCacheValid := true.B
        state := Mux(bFillForRun, s_req_fill_a, s_resp)
      }.otherwise {
        fillIdx := nextIdx
        state := s_req_fill_b
      }
    }
  }

  when(state === s_req_fill_a) {
    tlOut.a.valid := fillIdx < totalK
    tlOut.a.bits := getABits
    when(tlOut.a.fire) {
      incTLARead := true.B
      state := s_wait_fill_a
    }
  }

  when(state === s_wait_fill_a) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp, "FPGA-safe matmul expected AccessAckData on A-fill")
      assert(dSource === readTlSourceId.U(tlSourceIdxWidth.W), "FPGA-safe matmul received unexpected A-fill source")
      for (w <- 0 until wordsPerBeat) {
        when(aReadLaneValids(w)) {
          aBuf(aReadDestIdxs(w)) := readDataWords(w)
        }
      }
      val nextIdx = fillIdx + aWordsThisBeat
      when(nextIdx >= totalK) {
        fillIdx := 0.U
        computeIdx := 0.U
        incChunkStarted := true.B
        state := s_compute
      }.otherwise {
        fillIdx := nextIdx
        state := s_req_fill_a
      }
    }
  }

  when(state === s_compute) {
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        val product = aFixed(r).io.out * bFixed(c).io.out
        val sum = accum(r)(c) + product.asSInt
        accum(r)(c) := sum(accumPrec - 1, 0).asSInt
      }
    }
    when(computeIdx + 1.U >= totalK) {
      state := s_quantize
    }.otherwise {
      computeIdx := computeIdx + 1.U
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
      incTLCWrite := true.B
      state := s_wait_put
    }
  }

  when(state === s_wait_put) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsWriteAck, "FPGA-safe matmul expected AccessAck on C writeback")
      assert(dSource === writeTlSourceId.U(tlSourceIdxWidth.W), "FPGA-safe matmul received unexpected C write source")
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
    when(incPreloadCmd) {
      perfCounters(perfPreloadCmds) := perfCounters(perfPreloadCmds) + 1.U
    }
    when(incPreloadReuseHit) {
      perfCounters(perfPreloadReuseHits) := perfCounters(perfPreloadReuseHits) + 1.U
    }
    when(incChunkStarted) {
      perfCounters(perfChunksStarted) := perfCounters(perfChunksStarted) + 1.U
    }
    when(state === s_compute) {
      perfCounters(perfFeedCycles) := perfCounters(perfFeedCycles) + 1.U
    }
    when(state === s_quantize) {
      perfCounters(perfCaptureRows) := perfCounters(perfCaptureRows) + nRows.U
    }
    when(incTLBRead) {
      perfCounters(perfTLBReads) := perfCounters(perfTLBReads) + 1.U
    }
    when(incTLARead) {
      perfCounters(perfTLAReads) := perfCounters(perfTLAReads) + 1.U
    }
    when(incTLCWrite) {
      perfCounters(perfTLCWrites) := perfCounters(perfTLCWrites) + 1.U
    }
    when(state === s_wait_fill_b) {
      perfCounters(perfWaitFillBCycles) := perfCounters(perfWaitFillBCycles) + 1.U
    }
    when(state === s_wait_fill_a) {
      perfCounters(perfWaitFillACycles) := perfCounters(perfWaitFillACycles) + 1.U
    }
    when(state === s_compute) {
      perfCounters(perfWaitChunkOutCycles) := perfCounters(perfWaitChunkOutCycles) + 1.U
    }
    when(state === s_wait_put) {
      perfCounters(perfWaitPutCycles) := perfCounters(perfWaitPutCycles) + 1.U
    }
  }
}

class SystolicArrayFpgaSafeRoCC(
  opcodes: OpcodeSet,
  val precision: Int = 16,
  val nRows: Int = 4,
  val nCols: Int = 4,
  val maxK: Int = 256,
  val fixedPointFracBits: Int = 8,
  val accumBits: Int = 64,
  val numTLSourceIds: Int = 2
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new SystolicArrayFpgaSafeImpl(this)
  override val atlNode = TLClientNode(
    Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1(
      name = "SystolicArrayFpgaSafeRoCC",
      sourceId = IdRange(0, numTLSourceIds)
    ))))
  )
}

class SystolicArrayFpgaSafe8x8Impl(
  outer: SystolicArrayFpgaSafe8x8RoCC
)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters
  with HasL1CacheParameters {

  val cacheParams = tileParams.dcache.get

  private val precision = outer.precision // 16
  private val nRows = outer.nRows // 8
  private val nCols = outer.nCols // 8
  private val maxK = outer.maxK // 256
  private val fixedPointFracBits = outer.fixedPointFracBits // 8
  private val accumPrec = outer.accumBits // 64, keeps it simple for the ABI.
  private val tlSourceIds = outer.numTLSourceIds // 2, tags for different mem transactions
  private val applyRowSoftmax = outer.applyRowSoftmax // true, matmul or softmax
  private val softmaxIntPrecision = outer.softmaxIntPrecision
  private val softmaxFracPrecision = outer.softmaxFracPrecision
  private val softBitWidth = softmaxIntPrecision + softmaxFracPrecision
  private val lutEntries = 256 // Range of possible reciprocals
  private val lutBits = 64 // Precision of each reciprocal
  private val minBf16 = "hFF80".U(16.W) // -inf

  // sqrt(1/d_k)
  private val scoreFracBits = 2 * fixedPointFracBits // 16
  private val scoreScaleIntBits = 8

  private val kWidth = log2Ceil(maxK + 1) // 9, at which common dim
  private val outCount = nRows * nCols // 64, 8x8
  private val outputElemsPerWord = xLen / precision // 4, packed output
  // ceil_div(outCount, outputElemsPerWord)
  private val outputWordCount = (outCount + outputElemsPerWord - 1) / outputElemsPerWord // 16
  
  // Counter registers bitwidth.
  private val outIdxWidth = log2Ceil(outputWordCount + 1) // 5, which output words are okay
  private val rowIdxWidth = log2Ceil(nRows + 1) // 4, which softmax row we are at

  private val xlenBytes = xLen / 8 // 8
  private val beatBytes = cacheDataBits / 8 // 16
  private val wordsPerBeat = beatBytes / xlenBytes // 2
  private val laneWidth = math.max(1, log2Ceil(wordsPerBeat)) // 1
  private val beatOffsetBits = log2Ceil(beatBytes)
  private val tlSourceIdxWidth = math.max(1, log2Ceil(tlSourceIds))
  
  // tags
  private val readTlSourceId = 0
  private val writeTlSourceId = 1

  require(precision == 16, "FPGA-safe 8x8 matmul currently targets BF16 lanes")
  require(nRows == 8 && nCols == 8, "FPGA-safe 8x8 matmul targets one 8x8 tile")
  require(maxK > 0, "maxK must be positive")
  require(xLen == 64, s"8x8 software ABI writes uint64_t C words and expects RV64 xLen, got ${xLen}")
  require(cacheDataBits >= precision * nRows, s"8x8 A-record must fit in one cache beat (${cacheDataBits} < ${precision * nRows})")
  require(cacheDataBits >= precision * nCols, s"8x8 B-record must fit in one cache beat (${cacheDataBits} < ${precision * nCols})")
  require(cacheDataBits % xLen == 0, s"cacheDataBits must be a multiple of xLen (${cacheDataBits} % ${xLen})")
  require(isPow2(wordsPerBeat), s"wordsPerBeat must be a power of two, got ${wordsPerBeat}")
  require(accumPrec <= xLen, s"accumBits must fit in xLen (${accumPrec} > ${xLen})")
  require(outputElemsPerWord > 0, "xLen must hold at least one BF16 output")
  require(outputElemsPerWord == 4, "8x8 row-softmax packing expects four BF16 outputs per xLen word")
  require(tlSourceIds >= 2, "FPGA-safe 8x8 matmul uses one read source and one write source")
  require(softmaxFracPrecision >= log2Ceil(lutEntries) - 1, "softmax frac precision too small for LUT indexing")

  val aBuf = Reg(Vec(maxK, UInt(cacheDataBits.W))) // a vector
  val bBuf = Reg(Vec(maxK, UInt(cacheDataBits.W))) // b vector
  val accum = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.S(accumPrec.W)))))) // 0 init
  val packedStoreWords = Reg(Vec(outputWordCount, UInt(xLen.W))) // packed output
  val softRowIdx = RegInit(0.U(rowIdxWidth.W))
  val softLatchedScores = Reg(Vec(nCols, UInt(precision.W)))
  val softRowMax = RegInit(minBf16)
  val softInvSum = RegInit(0.U(softBitWidth.W))
  val scoreScaleBf16 = RegInit("h3F80".U(precision.W))

  val softVecMax = Module(new BFloat16VectorMax(nCols))
  val softVecSubs = Seq.fill(nCols)(Module(new BFloat16Sub))
  val softVecExps = Seq.fill(nCols)(Module(new BFloat16Exp))
  val softVecFixedPs = Seq.fill(nCols)(Module(new BFloat16ToFixed(softmaxIntPrecision, softmaxFracPrecision)))
  val softVecNormOut = Seq.fill(nCols)(Module(new UIntFixedToBFloat16(softBitWidth, softmaxFracPrecision)))
  val softVecFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  val softVecNormFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  val softProbBf16 = Wire(Vec(nCols, UInt(precision.W)))

  softVecMax.io.in := softLatchedScores.asUInt
  for (i <- 0 until nCols) {
    softVecSubs(i).io.in_1 := softLatchedScores(i)
    softVecSubs(i).io.in_2 := softRowMax
    softVecExps(i).io.in := softVecSubs(i).io.out
    softVecFixedPs(i).io.in := softVecExps(i).io.out
    softVecFixed(i) := softVecFixedPs(i).io.out

    val normFull = softVecFixed(i) * softInvSum
    softVecNormFixed(i) := (normFull >> softmaxFracPrecision)(softBitWidth - 1, 0)
    softVecNormOut(i).io.in := softVecNormFixed(i)
    softProbBf16(i) := softVecNormOut(i).io.out
  }

  val softVecSum = softVecFixed.reduceTree(_ + _)
  val softLutIndexBits = log2Ceil(lutEntries)
  val softLut = VecInit(Seq.tabulate(lutEntries) { i =>
    val mant = BigDecimal(1) + (BigDecimal(i) + BigDecimal(0.5)) / BigDecimal(lutEntries)
    val recip = BigDecimal(1) / mant
    val scaled = (recip * BigDecimal(2).pow(lutBits - 1)).setScale(0, RoundingMode.HALF_UP).toBigInt
    scaled.U(lutBits.W)
  })
  val softSumNonZero = softVecSum.orR
  val softMsb = (softBitWidth - 1).U - PriorityEncoder(Reverse(softVecSum))
  val softShift = Mux(softSumNonZero, softMsb - softmaxFracPrecision.U, 0.U)
  val softMant = softVecSum >> softShift
  val softLutIndex = softMant(softmaxFracPrecision, softmaxFracPrecision - softLutIndexBits + 1)
  val softLutVal = softLut(softLutIndex)
  val softScaleDown = lutBits - 1 - softmaxFracPrecision
  val softLutScaled =
    if (softScaleDown >= 0) (softLutVal >> softScaleDown) else (softLutVal << (-softScaleDown))
  val softInvRaw = softLutScaled >> softShift
  val softInvSumCandidate = Mux(softSumNonZero, softInvRaw.pad(softBitWidth)(softBitWidth - 1, 0), 0.U)

  val scoreScaleConv = Module(new BFloat16ToSIntFixed(scoreScaleIntBits, scoreFracBits))
  scoreScaleConv.io.in := scoreScaleBf16
  val scoreScaleFixed = scoreScaleConv.io.out

  // base pointers
  val aBase = RegInit(0.U(xLen.W))
  val bBase = RegInit(0.U(xLen.W))
  val cBase = RegInit(0.U(xLen.W))

  // Weight stationary.
  val totalK = RegInit(0.U(kWidth.W))
  val bCacheK = RegInit(0.U(kWidth.W))
  val bCacheValid = RegInit(false.B)
  val configured = RegInit(false.B)
  val bFillForRun = RegInit(false.B)

  val fillIdx = RegInit(0.U(kWidth.W))
  val computeIdx = RegInit(0.U(kWidth.W))
  val outIdx = RegInit(0.U(outIdxWidth.W))
  val respRd = RegInit(0.U(5.W))
  val respData = RegInit(0.U(xLen.W))

  val (s_idle :: s_req_fill_b :: s_wait_fill_b :: s_req_fill_a :: s_wait_fill_a ::
    s_compute :: s_softmax_load :: s_softmax_max :: s_softmax_sum :: s_softmax_norm ::
    s_quantize :: s_req_put :: s_wait_put :: s_resp :: Nil) = Enum(14)
  val state = RegInit(s_idle)

  private val numPerfCounters = 15
  private val perfCounterWidth = log2Ceil(numPerfCounters)
  private val perfBusyCycles = 0
  private val perfRunCmds = 1
  private val perfPreloadCmds = 2
  private val perfPreloadReuseHits = 3
  private val perfChunksStarted = 4
  private val perfFeedCycles = 5
  private val perfCaptureRows = 6
  private val perfTLBReads = 7
  private val perfTLAReads = 8
  private val perfTLCWrites = 9
  private val perfWaitFillBCycles = 10
  private val perfWaitFillACycles = 11
  private val perfLoadWeightCycles = 12
  private val perfWaitChunkOutCycles = 13
  private val perfWaitPutCycles = 14
  val perfCounters = RegInit(VecInit(Seq.fill(numPerfCounters)(0.U(xLen.W))))

  val clearPerfCounters = WireDefault(false.B)
  val incRunCmd = WireDefault(false.B)
  val incPreloadCmd = WireDefault(false.B)
  val incPreloadReuseHit = WireDefault(false.B)
  val incChunkStarted = WireDefault(false.B)
  val incTLBRead = WireDefault(false.B)
  val incTLARead = WireDefault(false.B)
  val incTLCWrite = WireDefault(false.B)

  // Did we recieve the data?
  // Did the memory acknowledged our output?
  // Is the tlSrc matching the tag?
  val (tlOut, edgesOut) = outer.atlNode.out(0)
  val dHasData = edgesOut.hasData(tlOut.d.bits)
  val dSource = tlOut.d.bits.source(tlSourceIdxWidth - 1, 0)
  val dIsReadResp = dHasData && tlOut.d.bits.opcode === TLMessages.AccessAckData
  val dIsWriteAck = !dHasData && tlOut.d.bits.opcode === TLMessages.AccessAck

  def beatAddr(base: UInt, idx: UInt): UInt = base + idx * beatBytes.U
  def wordAddr(base: UInt, idx: UInt): UInt = base + idx * xlenBytes.U

  val beatLgSize = beatOffsetBits.U
  val getBBits = edgesOut.Get(
    fromSource = readTlSourceId.U(tlSourceIdxWidth.W),
    toAddress = beatAddr(bBase, fillIdx),
    lgSize = beatLgSize
  )._2
  val getABits = edgesOut.Get(
    fromSource = readTlSourceId.U(tlSourceIdxWidth.W),
    toAddress = beatAddr(aBase, fillIdx),
    lgSize = beatLgSize
  )._2

  val writeAddr = wordAddr(cBase, outIdx)
  val writeLane = writeAddr(beatOffsetBits - 1, log2Ceil(xlenBytes))
  val writeBeatAddr = writeAddr & ~((beatBytes - 1).U(xLen.W))
  val writeLaneWordIdxs = Wire(Vec(wordsPerBeat, UInt(outIdxWidth.W)))
  val writeLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    val laneAfterStart = w.U(laneWidth.W) >= writeLane
    val laneDelta = Mux(laneAfterStart, w.U(laneWidth.W) - writeLane, 0.U(laneWidth.W))
    writeLaneWordIdxs(w) := outIdx + laneDelta
    writeLaneValids(w) := laneAfterStart && writeLaneWordIdxs(w) < outputWordCount.U
  }
  val writeWordsThisBeat = PopCount(writeLaneValids)
  val putDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    putDataWords(w) := Mux(writeLaneValids(w), packedStoreWords(writeLaneWordIdxs(w)), 0.U)
  }
  val putMaskBytes = Wire(Vec(beatBytes, Bool()))
  for (byte <- 0 until beatBytes) {
    putMaskBytes(byte) := writeLaneValids(byte / xlenBytes)
  }
  val putBits = edgesOut.Put(
    fromSource = writeTlSourceId.U(tlSourceIdxWidth.W),
    toAddress = writeBeatAddr,
    lgSize = beatLgSize,
    data = putDataWords.asUInt,
    mask = putMaskBytes.asUInt
  )._2

  // read from buffers
  val aLanes = Wire(Vec(nRows, UInt(precision.W)))
  val bLanes = Wire(Vec(nCols, UInt(precision.W)))
  for (r <- 0 until nRows) {
    aLanes(r) := aBuf(computeIdx)(precision * (r + 1) - 1, precision * r)
  }
  for (c <- 0 until nCols) {
    bLanes(c) := bBuf(computeIdx)(precision * (c + 1) - 1, precision * c)
  }

  // convert buffers bf16 data to fixed
  val aFixed = Seq.fill(nRows)(Module(new BFloat16ToSIntFixed(intBits = precision - fixedPointFracBits, fracBits = fixedPointFracBits)))
  val bFixed = Seq.fill(nCols)(Module(new BFloat16ToSIntFixed(intBits = precision - fixedPointFracBits, fracBits = fixedPointFracBits)))
  for (r <- 0 until nRows) {
    aFixed(r).io.in := aLanes(r)
  }
  for (c <- 0 until nCols) {
    bFixed(c).io.in := bLanes(c)
  }

  val bf16Out = Wire(Vec(outCount, UInt(precision.W)))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
                                                // 64    , 16
      val conv = Module(new SIntFixedToBFloat16(accumPrec, 2 * fixedPointFracBits))
      val scoreForConv = Wire(SInt(accumPrec.W))
      if (applyRowSoftmax) {
        val scaledWide = (accum(r)(c) * scoreScaleFixed).asSInt
        val scaledShifted = (scaledWide >> scoreFracBits).asSInt
        scoreForConv := scaledShifted(accumPrec - 1, 0).asSInt
      } else {
        scoreForConv := accum(r)(c)
      }
      conv.io.in := scoreForConv
      bf16Out(r * nCols + c) := conv.io.out
    }
  }

  val packedStoreWordsWire = Wire(Vec(outputWordCount, UInt(xLen.W)))
  for (w <- 0 until outputWordCount) {
    val packedElems = Wire(Vec(outputElemsPerWord, UInt(precision.W)))
    for (e <- 0 until outputElemsPerWord) {
      val idx = w * outputElemsPerWord + e
      if (idx < outCount) {
        packedElems(e) := bf16Out(idx)
      } else {
        packedElems(e) := 0.U
      }
    }
    packedStoreWordsWire(w) := packedElems.asUInt
  }

  tlOut.a.valid := false.B
  tlOut.a.bits := getABits
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

    when(funct === 0.U) {
      aBase := io.cmd.bits.rs1
      bBase := io.cmd.bits.rs2
      configured := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 1.U) {
      incRunCmd := true.B
      val requestedK = io.cmd.bits.rs2
      cBase := io.cmd.bits.rs1
      totalK := requestedK(kWidth - 1, 0)
      bCacheK := requestedK(kWidth - 1, 0)
      fillIdx := 0.U
      computeIdx := 0.U
      outIdx := 0.U
      softRowIdx := 0.U
      bFillForRun := true.B
      bCacheValid := false.B
      for (r <- 0 until nRows) {
        for (c <- 0 until nCols) {
          accum(r)(c) := 0.S
        }
      }
      respData := Mux(!configured, 1.U, Mux(requestedK > maxK.U, 2.U, 0.U))
      val doneState = if (applyRowSoftmax) s_softmax_load else s_quantize
      state := Mux(
        !configured || requestedK > maxK.U,
        s_resp,
        Mux(requestedK === 0.U, doneState, s_req_fill_b)
      )
    }.elsewhen(funct === 3.U) {
      incPreloadCmd := true.B
      val requestedK = io.cmd.bits.rs2
      bBase := io.cmd.bits.rs1
      bCacheK := requestedK(kWidth - 1, 0)
      fillIdx := 0.U
      bFillForRun := false.B
      bCacheValid := requestedK === 0.U
      configured := true.B
      respData := Mux(requestedK > maxK.U, 2.U, 0.U)
      state := Mux(requestedK > maxK.U || requestedK === 0.U, s_resp, s_req_fill_b)
    }.elsewhen(funct === 6.U) {
      incRunCmd := true.B
      incPreloadReuseHit := bCacheValid
      aBase := io.cmd.bits.rs1
      cBase := io.cmd.bits.rs2
      totalK := bCacheK
      fillIdx := 0.U
      computeIdx := 0.U
      outIdx := 0.U
      softRowIdx := 0.U
      for (r <- 0 until nRows) {
        for (c <- 0 until nCols) {
          accum(r)(c) := 0.S
        }
      }
      respData := Mux(!bCacheValid, 3.U, 0.U)
      val doneState = if (applyRowSoftmax) s_softmax_load else s_quantize
      state := Mux(
        !bCacheValid,
        s_resp,
        Mux(bCacheK === 0.U, doneState, s_req_fill_a)
      )
    }.elsewhen(funct === 2.U) {
      respData := Cat(0.U((xLen - 2).W), state =/= s_idle, state === s_resp)
      state := s_resp
    }.elsewhen(funct === 4.U) {
      val counterSel = io.cmd.bits.rs1
      val counterIdx = counterSel(perfCounterWidth - 1, 0)
      respData := Mux(
        counterSel < numPerfCounters.U,
        perfCounters(counterIdx),
        "hBAD0BAD0BAD0BAD0".U(xLen.W)
      )
      state := s_resp
    }.elsewhen(funct === 5.U) {
      clearPerfCounters := true.B
      respData := 0.U
      state := s_resp
    }.elsewhen(funct === 7.U) {
      scoreScaleBf16 := io.cmd.bits.rs1(precision - 1, 0)
      respData := 0.U
      state := s_resp
    }.otherwise {
      respData := "hDEAD".U
      state := s_resp
    }
  }

  when(state === s_req_fill_b) {
    tlOut.a.valid := fillIdx < bCacheK
    tlOut.a.bits := getBBits
    when(tlOut.a.fire) {
      incTLBRead := true.B
      state := s_wait_fill_b
    }
  }

  when(state === s_wait_fill_b) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp, "FPGA-safe 8x8 matmul expected AccessAckData on B-fill")
      assert(dSource === readTlSourceId.U(tlSourceIdxWidth.W), "FPGA-safe 8x8 matmul received unexpected B-fill source")
      bBuf(fillIdx) := tlOut.d.bits.data(cacheDataBits - 1, 0)
      val nextIdx = fillIdx + 1.U
      when(nextIdx >= bCacheK) {
        fillIdx := 0.U
        bCacheValid := true.B
        state := Mux(bFillForRun, s_req_fill_a, s_resp)
      }.otherwise {
        fillIdx := nextIdx
        state := s_req_fill_b
      }
    }
  }

  when(state === s_req_fill_a) {
    tlOut.a.valid := fillIdx < totalK
    tlOut.a.bits := getABits
    when(tlOut.a.fire) {
      incTLARead := true.B
      state := s_wait_fill_a
    }
  }

  when(state === s_wait_fill_a) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsReadResp, "FPGA-safe 8x8 matmul expected AccessAckData on A-fill")
      assert(dSource === readTlSourceId.U(tlSourceIdxWidth.W), "FPGA-safe 8x8 matmul received unexpected A-fill source")
      aBuf(fillIdx) := tlOut.d.bits.data(cacheDataBits - 1, 0)
      val nextIdx = fillIdx + 1.U
      when(nextIdx >= totalK) {
        fillIdx := 0.U
        computeIdx := 0.U
        incChunkStarted := true.B
        state := s_compute
      }.otherwise {
        fillIdx := nextIdx
        state := s_req_fill_a
      }
    }
  }

  when(state === s_compute) {
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        val product = aFixed(r).io.out * bFixed(c).io.out
        val sum = accum(r)(c) + product.asSInt
        accum(r)(c) := sum(accumPrec - 1, 0).asSInt
      }
    }
    when(computeIdx + 1.U >= totalK) {
      if (applyRowSoftmax) {
        state := s_softmax_load
      } else {
        state := s_quantize
      }
    }.otherwise {
      computeIdx := computeIdx + 1.U
    }
  }

  when(state === s_softmax_load) {
    for (c <- 0 until nCols) {
      softLatchedScores(c) := bf16Out(softRowIdx * nCols.U + c.U)
    }
    state := s_softmax_max
  }

  when(state === s_softmax_max) {
    softRowMax := softVecMax.io.out
    state := s_softmax_sum
  }

  when(state === s_softmax_sum) {
    softInvSum := softInvSumCandidate
    state := s_softmax_norm
  }

  when(state === s_softmax_norm) {
    val rowWordBase = softRowIdx << 1
    val lo = Wire(Vec(4, UInt(precision.W)))
    val hi = Wire(Vec(4, UInt(precision.W)))
    for (i <- 0 until 4) {
      lo(i) := softProbBf16(i)
      hi(i) := softProbBf16(i + 4)
    }
    packedStoreWords(rowWordBase) := lo.asUInt
    packedStoreWords(rowWordBase + 1.U) := hi.asUInt
    when(softRowIdx + 1.U >= nRows.U) {
      outIdx := 0.U
      state := s_req_put
    }.otherwise {
      softRowIdx := softRowIdx + 1.U
      state := s_softmax_load
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
      incTLCWrite := true.B
      state := s_wait_put
    }
  }

  when(state === s_wait_put) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      assert(dIsWriteAck, "FPGA-safe 8x8 matmul expected AccessAck on C writeback")
      assert(dSource === writeTlSourceId.U(tlSourceIdxWidth.W), "FPGA-safe 8x8 matmul received unexpected C write source")
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
    when(incPreloadCmd) {
      perfCounters(perfPreloadCmds) := perfCounters(perfPreloadCmds) + 1.U
    }
    when(incPreloadReuseHit) {
      perfCounters(perfPreloadReuseHits) := perfCounters(perfPreloadReuseHits) + 1.U
    }
    when(incChunkStarted) {
      perfCounters(perfChunksStarted) := perfCounters(perfChunksStarted) + 1.U
    }
    when(state === s_compute) {
      perfCounters(perfFeedCycles) := perfCounters(perfFeedCycles) + 1.U
    }
    when(state === s_quantize) {
      perfCounters(perfCaptureRows) := perfCounters(perfCaptureRows) + nRows.U
    }
    when(incTLBRead) {
      perfCounters(perfTLBReads) := perfCounters(perfTLBReads) + 1.U
    }
    when(incTLARead) {
      perfCounters(perfTLAReads) := perfCounters(perfTLAReads) + 1.U
    }
    when(incTLCWrite) {
      perfCounters(perfTLCWrites) := perfCounters(perfTLCWrites) + 1.U
    }
    when(state === s_wait_fill_b) {
      perfCounters(perfWaitFillBCycles) := perfCounters(perfWaitFillBCycles) + 1.U
    }
    when(state === s_wait_fill_a) {
      perfCounters(perfWaitFillACycles) := perfCounters(perfWaitFillACycles) + 1.U
    }
    when(state === s_compute) {
      perfCounters(perfWaitChunkOutCycles) := perfCounters(perfWaitChunkOutCycles) + 1.U
    }
    when(state === s_wait_put) {
      perfCounters(perfWaitPutCycles) := perfCounters(perfWaitPutCycles) + 1.U
    }
  }
}

class SystolicArrayFpgaSafe8x8RoCC(
  opcodes: OpcodeSet,
  val precision: Int = 16,
  val nRows: Int = 8,
  val nCols: Int = 8,
  val maxK: Int = 256,
  val fixedPointFracBits: Int = 8,
  val accumBits: Int = 64,
  val numTLSourceIds: Int = 2,
  val applyRowSoftmax: Boolean = false,
  val softmaxIntPrecision: Int = 12,
  val softmaxFracPrecision: Int = 20,
  val clientName: String = "SystolicArrayFpgaSafe8x8RoCC"
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new SystolicArrayFpgaSafe8x8Impl(this)
  override val atlNode = TLClientNode(
    Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1(
      name = clientName,
      sourceId = IdRange(0, numTLSourceIds)
    ))))
  )
}
