package toyrocc

import chisel3._
import chisel3.util._

import freechips.rocketchip.diplomacy.IdRange
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink.{TLClientNode, TLMasterParameters, TLMasterPortParameters}
import org.chipsalliance.cde.config.Parameters

class WeightStationaryArrayCoreIO(
  precision: Int,
  nRows: Int,
  nCols: Int,
  accumPrec: Int
) extends Bundle {
  val loadWeights = Input(Bool())
  val clearWeights = Input(Bool())
  val weightTile = Input(Vec(nRows, Vec(nCols, UInt(precision.W))))

  val inValid = Input(Bool())
  val inVec = Input(Vec(nRows, UInt(precision.W)))

  val outValid = Output(Bool())
  val outVec = Output(Vec(nCols, UInt(accumPrec.W)))
}

abstract class WeightStationaryArrayCoreBase(
  precision: Int,
  nRows: Int,
  nCols: Int,
  accumPrec: Int
) extends Module {
  val io = IO(new WeightStationaryArrayCoreIO(
    precision = precision,
    nRows = nRows,
    nCols = nCols,
    accumPrec = accumPrec
  ))
}

class ProcessingElementWS(inPrec: Int = 16, accumPrec: Int = 32) extends Module {
  val io = IO(new Bundle {
    val loadWeight = Input(Bool())
    val clearWeight = Input(Bool())
    val weightIn = Input(UInt(inPrec.W))

    val inElem = Input(UInt(inPrec.W))
    val inPartialSum = Input(UInt(accumPrec.W))

    val outElem = Output(UInt(inPrec.W))
    val outPartialSum = Output(UInt(accumPrec.W))
  })

  val weight = RegInit(0.U(inPrec.W))

  when(io.clearWeight) {
    weight := 0.U
  }.elsewhen(io.loadWeight) {
    weight := io.weightIn
  }

  io.outElem := RegNext(io.inElem, 0.U)
  io.outPartialSum := RegNext(io.inPartialSum + io.inElem * weight, 0.U)
}

class ProcessingElementWSFixed(inPrec: Int = 16, accumPrec: Int = 64) extends Module {
  val io = IO(new Bundle {
    val loadWeight = Input(Bool())
    val clearWeight = Input(Bool())
    val weightIn = Input(SInt(inPrec.W))

    val inElem = Input(SInt(inPrec.W))
    val inPartialSum = Input(SInt(accumPrec.W))

    val outElem = Output(SInt(inPrec.W))
    val outPartialSum = Output(SInt(accumPrec.W))
  })

  val weight = RegInit(0.S(inPrec.W))

  when(io.clearWeight) {
    weight := 0.S
  }.elsewhen(io.loadWeight) {
    weight := io.weightIn
  }

  val mac = io.inPartialSum + (io.inElem * weight).asSInt
  io.outElem := RegNext(io.inElem, 0.S(inPrec.W))
  io.outPartialSum := RegNext(mac(accumPrec - 1, 0).asSInt, 0.S(accumPrec.W))
}

class MeshWS(
  precision: Int = 16,
  nRows: Int = 4,
  nCols: Int = 4,
  accumPrec: Int = 32
) extends Module {
  val io = IO(new Bundle {
    val loadWeights = Input(Bool())
    val clearWeights = Input(Bool())
    val weights = Input(Vec(nRows, Vec(nCols, UInt(precision.W))))

    val inElems = Input(Vec(nRows, UInt(precision.W)))
    val outPartialSums = Output(Vec(nCols, UInt(accumPrec.W)))
  })

  val pes = Seq.fill(nRows)(Seq.fill(nCols)(Module(new ProcessingElementWS(precision, accumPrec))))

  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      pes(r)(c).io.loadWeight := io.loadWeights
      pes(r)(c).io.clearWeight := io.clearWeights
      pes(r)(c).io.weightIn := io.weights(r)(c)
    }
  }

  pes(0)(0).io.inElem := io.inElems(0)
  pes(0)(0).io.inPartialSum := 0.U

  for (r <- 1 until nRows) {
    pes(r)(0).io.inElem := io.inElems(r)
    pes(r)(0).io.inPartialSum := pes(r - 1)(0).io.outPartialSum
  }

  for (c <- 1 until nCols) {
    pes(0)(c).io.inElem := pes(0)(c - 1).io.outElem
    pes(0)(c).io.inPartialSum := 0.U
  }

  for (r <- 1 until nRows) {
    for (c <- 1 until nCols) {
      pes(r)(c).io.inElem := pes(r)(c - 1).io.outElem
      pes(r)(c).io.inPartialSum := pes(r - 1)(c).io.outPartialSum
    }
  }

  for (c <- 0 until nCols) {
    io.outPartialSums(c) := pes(nRows - 1)(c).io.outPartialSum
  }
}

class MeshWSFixed(
  precision: Int = 16,
  nRows: Int = 4,
  nCols: Int = 4,
  accumPrec: Int = 64
) extends Module {
  val io = IO(new Bundle {
    val loadWeights = Input(Bool())
    val clearWeights = Input(Bool())
    val weights = Input(Vec(nRows, Vec(nCols, SInt(precision.W))))

    val inElems = Input(Vec(nRows, SInt(precision.W)))
    val outPartialSums = Output(Vec(nCols, SInt(accumPrec.W)))
  })

  val pes = Seq.fill(nRows)(Seq.fill(nCols)(Module(new ProcessingElementWSFixed(precision, accumPrec))))

  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      pes(r)(c).io.loadWeight := io.loadWeights
      pes(r)(c).io.clearWeight := io.clearWeights
      pes(r)(c).io.weightIn := io.weights(r)(c)
    }
  }

  pes(0)(0).io.inElem := io.inElems(0)
  pes(0)(0).io.inPartialSum := 0.S(accumPrec.W)

  for (r <- 1 until nRows) {
    pes(r)(0).io.inElem := io.inElems(r)
    pes(r)(0).io.inPartialSum := pes(r - 1)(0).io.outPartialSum
  }

  for (c <- 1 until nCols) {
    pes(0)(c).io.inElem := pes(0)(c - 1).io.outElem
    pes(0)(c).io.inPartialSum := 0.S(accumPrec.W)
  }

  for (r <- 1 until nRows) {
    for (c <- 1 until nCols) {
      pes(r)(c).io.inElem := pes(r)(c - 1).io.outElem
      pes(r)(c).io.inPartialSum := pes(r - 1)(c).io.outPartialSum
    }
  }

  for (c <- 0 until nCols) {
    io.outPartialSums(c) := pes(nRows - 1)(c).io.outPartialSum
  }
}

class WeightStationaryArrayCoreInt(
  precision: Int = 16,
  nRows: Int = 4,
  nCols: Int = 4,
  accumPrec: Int = 32
) extends WeightStationaryArrayCoreBase(
  precision = precision,
  nRows = nRows,
  nCols = nCols,
  accumPrec = accumPrec
) {
  require(nRows == nCols, "Weight-stationary core currently assumes a square array")
  // Time until 1st output
  private val outputLatency = 2 * nRows - 1

  val mesh = Module(new MeshWS(
    precision = precision,
    nRows = nRows,
    nCols = nCols,
    accumPrec = accumPrec
  ))

  mesh.io.loadWeights := io.loadWeights
  mesh.io.clearWeights := io.clearWeights
  mesh.io.weights := io.weightTile

  val skewedInputs = Wire(Vec(nRows, UInt(precision.W)))
  for (r <- 0 until nRows) {
    val inElem = Mux(io.inValid, io.inVec(r), 0.U(precision.W))
    skewedInputs(r) := ShiftRegister(inElem, r)
  }
  mesh.io.inElems := skewedInputs

  for (c <- 0 until nCols) {
    io.outVec(c) := ShiftRegister(mesh.io.outPartialSums(c), nCols - 1 - c)
  }
  io.outValid := ShiftRegister(io.inValid, outputLatency)
}

class WeightStationaryArrayCoreBF16Fixed(
  precision: Int = 16,
  nRows: Int = 4,
  nCols: Int = 4,
  accumPrec: Int = 64,
  fracBits: Int = 8
) extends WeightStationaryArrayCoreBase(
  precision = precision,
  nRows = nRows,
  nCols = nCols,
  accumPrec = accumPrec
) {
  require(nRows == nCols, "Weight-stationary core currently assumes a square array")
  require(precision >= 2, "fixed-point precision must be at least 2 bits")
  require(fracBits >= 0 && fracBits < precision, s"fracBits must be in [0, precision), got $fracBits")

  private val outputLatency = 2 * nRows - 1

  val mesh = Module(new MeshWSFixed(
    precision = precision,
    nRows = nRows,
    nCols = nCols,
    accumPrec = accumPrec
  ))

  val weightFixed = Wire(Vec(nRows, Vec(nCols, SInt(precision.W))))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      val conv = Module(new BFloat16ToSIntFixed(intBits = precision - fracBits, fracBits = fracBits))
      conv.io.in := io.weightTile(r)(c)
      weightFixed(r)(c) := conv.io.out
    }
  }

  val inputFixed = Wire(Vec(nRows, SInt(precision.W)))
  for (r <- 0 until nRows) {
    val conv = Module(new BFloat16ToSIntFixed(intBits = precision - fracBits, fracBits = fracBits))
    conv.io.in := io.inVec(r)
    inputFixed(r) := conv.io.out
  }

  mesh.io.loadWeights := io.loadWeights
  mesh.io.clearWeights := io.clearWeights
  mesh.io.weights := weightFixed

  val skewedInputs = Wire(Vec(nRows, SInt(precision.W)))
  for (r <- 0 until nRows) {
    val inElem = Mux(io.inValid, inputFixed(r), 0.S(precision.W))
    skewedInputs(r) := ShiftRegister(inElem, r)
  }
  mesh.io.inElems := skewedInputs

  // Skew output
  for (c <- 0 until nCols) {
    io.outVec(c) := ShiftRegister(mesh.io.outPartialSums(c), nCols - 1 - c).asUInt
  }
  io.outValid := ShiftRegister(io.inValid, outputLatency)
}

class SystolicArrayWSImpl(
  outer: SystolicArrayWSRoCC
)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters
  with HasL1CacheParameters {

  val cacheParams = tileParams.dcache.get

  private val precision = outer.precision
  private val nRows = outer.nRows
  private val nCols = outer.nCols
  private val maxK = outer.maxK
  private val useBFloat16Input = outer.useBFloat16Input
  private val useBFloat16Output = outer.useBFloat16Output
  private val fixedPointFracBits = outer.fixedPointFracBits
  private val chunkKMax = nRows
  private val kWidth = log2Ceil(maxK + 1)
  private val cacheIdxWidth = log2Ceil(maxK + 1)
  private val chunkWidth = log2Ceil(chunkKMax + 1)
  private val accumPrec = outer.accumBits
  private val outCount = nRows * nCols
  private val outputElemsPerWord = if (useBFloat16Output) (xLen / precision) else 1
  private val outputWordCount = (outCount + outputElemsPerWord - 1) / outputElemsPerWord
  private val outIdxWidth = log2Ceil(outputWordCount + 1)
  private val xlenBytes = xLen / 8
  private val lgXlenBytes = log2Ceil(xlenBytes)
  private val beatBytes = cacheDataBits / 8
  private val wordsPerBeat = beatBytes / xlenBytes
  private val tlSourceIds = outer.numTLSourceIds
  private val tlSourceIdxWidth = math.max(1, log2Ceil(tlSourceIds))
  private val laneWidth = math.max(1, log2Ceil(wordsPerBeat))
  private val beatOffsetBits = log2Ceil(beatBytes)

  require(isPow2(xlenBytes), s"xLen bytes must be a power of two, got ${xlenBytes}")
  require(cacheDataBits % xLen == 0, s"cacheDataBits must be a multiple of xLen (${cacheDataBits} % ${xLen})")
  require(wordsPerBeat > 0, "wordsPerBeat must be > 0")
  require(isPow2(wordsPerBeat), s"wordsPerBeat must be a power of two, got ${wordsPerBeat}")
  require(tlSourceIds > 0, s"numTLSourceIds must be > 0, got ${tlSourceIds}")
  require(nRows > 0 && nCols > 0, "mesh dimensions must be > 0")
  require(nRows == nCols, "weight-stationary array currently requires a square mesh")
  require(maxK > 0, "maxK must be > 0")
  require(!useBFloat16Input || precision == 16, s"BF16 WS path expects 16-bit lanes, got precision=$precision")
  require(!useBFloat16Output || useBFloat16Input, "BF16 output currently requires the BF16-input fixed-point core")
  require(!useBFloat16Input || fixedPointFracBits < precision, s"fixedPointFracBits must be < precision, got $fixedPointFracBits >= $precision")
  require(precision * nRows <= xLen, s"A-stream word exceeds xLen (${precision * nRows} > ${xLen})")
  require(precision * nCols <= xLen, s"B-stream word exceeds xLen (${precision * nCols} > ${xLen})")
  require(accumPrec <= xLen, s"output element width exceeds xLen (${accumPrec} > ${xLen})")
  require(!useBFloat16Output || (xLen % precision == 0), s"xLen (${xLen}) must be a multiple of BF16 output width (${precision})")

  val core: WeightStationaryArrayCoreBase = if (useBFloat16Input) {
    Module(new WeightStationaryArrayCoreBF16Fixed(
      precision = precision,
      nRows = nRows,
      nCols = nCols,
      accumPrec = accumPrec,
      fracBits = fixedPointFracBits
    ))
  } else {
    Module(new WeightStationaryArrayCoreInt(
      precision = precision,
      nRows = nRows,
      nCols = nCols,
      accumPrec = accumPrec
    ))
  }

  val aChunkBufs = Reg(Vec(2, Vec(chunkKMax, UInt(xLen.W))))
  val bCacheBuf = Reg(Vec(maxK, UInt(xLen.W)))
  val accumTile = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.U(accumPrec.W))))))

  val weightTile = Wire(Vec(nRows, Vec(nCols, UInt(precision.W))))
  val currentInVec = Wire(Vec(nRows, UInt(precision.W)))
  val aChunkLanes = Wire(Vec(chunkKMax, Vec(nRows, UInt(precision.W))))

  val flatOut = Wire(Vec(outCount, UInt(accumPrec.W)))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      flatOut(r * nCols + c) := accumTile(r)(c)
    }
  }

  val packedBFloatOut = if (useBFloat16Output) Some(Wire(Vec(outCount, UInt(precision.W)))) else None
  val packedStoreWordsWire = if (useBFloat16Output) Some(Wire(Vec(outputWordCount, UInt(xLen.W)))) else None
  val packedStoreWordsReg = if (useBFloat16Output) Some(Reg(Vec(outputWordCount, UInt(xLen.W)))) else None

  if (useBFloat16Output) {
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        val conv = Module(new SIntFixedToBFloat16(accumPrec, 2 * fixedPointFracBits))
        conv.io.in := accumTile(r)(c).asSInt
        packedBFloatOut.get(r * nCols + c) := conv.io.out
      }
    }

    for (w <- 0 until outputWordCount) {
      val packedElems = Wire(Vec(outputElemsPerWord, UInt(precision.W)))
      for (e <- 0 until outputElemsPerWord) {
        val idx = w * outputElemsPerWord + e
        if (idx < outCount) {
          packedElems(e) := packedBFloatOut.get(idx)
        } else {
          packedElems(e) := 0.U
        }
      }
      packedStoreWordsWire.get(w) := packedElems.asUInt
    }
  }

  val s_idle :: s_req_fill_b_cache :: s_wait_fill_b_cache :: s_prep_chunk :: s_req_fill_a :: s_wait_fill_a :: s_load_weights :: s_feed_rows :: s_wait_chunk_out :: s_quantize_out :: s_req_put :: s_wait_put :: s_resp :: Nil = Enum(13)
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

  val aBase = RegInit(0.U(xLen.W))
  val bBase = RegInit(0.U(xLen.W))
  val cBase = RegInit(0.U(xLen.W))
  val configured = RegInit(false.B)

  val totalK = RegInit(0.U(xLen.W))
  val kRemaining = RegInit(0.U(xLen.W))
  val kBaseIdx = RegInit(0.U(xLen.W))
  val chunkK = RegInit(0.U(chunkWidth.W))
  val fillIdx = RegInit(0.U(chunkWidth.W))
  val feedRowIdx = RegInit(0.U(chunkWidth.W))
  val captureRowIdx = RegInit(0.U(chunkWidth.W))
  val outIdx = RegInit(0.U(outIdxWidth.W))
  val bCacheFillIdx = RegInit(0.U(cacheIdxWidth.W))
  val bIssueIdx = RegInit(0.U(cacheIdxWidth.W))
  val bCacheValid = RegInit(false.B)
  val pinnedWeightsValid = RegInit(false.B)
  val bCacheK = RegInit(0.U(xLen.W))
  val cachedBBase = RegInit(0.U(xLen.W))
  val bFillForRun = RegInit(false.B)
  val bFillInflight = RegInit(VecInit(Seq.fill(tlSourceIds)(false.B)))
  val bFillInflightIdx = Reg(Vec(tlSourceIds, UInt(cacheIdxWidth.W)))

  val respRd = RegInit(0.U(5.W))
  val respData = RegInit(0.U(xLen.W))

  val activeABufSel = RegInit(0.U(1.W))
  val prefetchedABufSel = RegInit(1.U(1.W))
  val prefetchedValid = RegInit(false.B)
  val prefetchedBaseIdx = RegInit(0.U(xLen.W))
  val prefetchedChunkK = RegInit(0.U(chunkWidth.W))
  val prefetchFillIdx = RegInit(0.U(chunkWidth.W))

  val pf_idle :: pf_req_fill_a :: pf_wait_fill_a :: Nil = Enum(3)
  val prefetchState = RegInit(pf_idle)

  for (wr <- 0 until nRows) {
    val weightIdx = (kBaseIdx + wr.U)(cacheIdxWidth - 1, 0)
    val weightWord = bCacheBuf(weightIdx)
    for (wc <- 0 until nCols) {
      weightTile(wr)(wc) := Mux(
        wr.U < chunkK,
        weightWord(precision * (wc + 1) - 1, precision * wc),
        0.U
      )
    }
  }

  for (k <- 0 until chunkKMax) {
    for (r <- 0 until nRows) {
      aChunkLanes(k)(r) := aChunkBufs(activeABufSel)(k)(precision * (r + 1) - 1, precision * r)
    }
  }

  val safeFeedRowIdx = WireDefault(0.U(chunkWidth.W))
  when(feedRowIdx < nRows.U) {
    safeFeedRowIdx := feedRowIdx
  }

  for (i <- 0 until nRows) {
    currentInVec(i) := Mux(
      i.U < chunkK,
      aChunkLanes(i)(safeFeedRowIdx),
      0.U
    )
  }

  core.io.loadWeights := false.B
  core.io.clearWeights := false.B
  core.io.weightTile := weightTile
  core.io.inValid := false.B
  core.io.inVec := currentInVec

  val (tlOut, edgesOut) = outer.atlNode.out(0)

  val absFillIdx = kBaseIdx + fillIdx
  val aAddr = aBase + absFillIdx * xlenBytes.U
  val writeAddr = cBase + outIdx * xlenBytes.U
  val aLane = if (wordsPerBeat > 1) aAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  val writeLane = if (wordsPerBeat > 1) writeAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)

  val aBeatBaseIdx = if (wordsPerBeat > 1) (absFillIdx >> laneWidth) << laneWidth else absFillIdx
  val writeBeatBaseIdx = if (wordsPerBeat > 1) (outIdx >> laneWidth) << laneWidth else outIdx

  val aBeatAddr = aBase + aBeatBaseIdx * xlenBytes.U
  val writeBeatAddr = cBase + writeBeatBaseIdx * xlenBytes.U

  val readLgSize = if (wordsPerBeat > 1) beatOffsetBits.U else lgXlenBytes.U

  val getABits = edgesOut.Get(fromSource = 0.U, toAddress = aBeatAddr, lgSize = readLgSize)._2

  val prefetchAAddr = aBase + (prefetchedBaseIdx + prefetchFillIdx) * xlenBytes.U
  val prefetchALane = if (wordsPerBeat > 1) prefetchAAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  val prefetchABeatBaseIdx = if (wordsPerBeat > 1) ((prefetchedBaseIdx + prefetchFillIdx) >> laneWidth) << laneWidth else (prefetchedBaseIdx + prefetchFillIdx)
  val prefetchABeatAddr = aBase + prefetchABeatBaseIdx * xlenBytes.U
  val getPrefetchABits = edgesOut.Get(fromSource = 0.U, toAddress = prefetchABeatAddr, lgSize = readLgSize)._2

  val aReadDestIdxs = Wire(Vec(wordsPerBeat, UInt(chunkWidth.W)))
  val aReadLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    if (wordsPerBeat > 1) {
      val laneAfterStart = w.U(laneWidth.W) >= aLane
      val laneDelta = Mux(laneAfterStart, w.U(laneWidth.W) - aLane, 0.U(laneWidth.W))
      aReadDestIdxs(w) := fillIdx + laneDelta
      aReadLaneValids(w) := laneAfterStart && (aReadDestIdxs(w) < chunkK)
    } else {
      aReadDestIdxs(w) := fillIdx
      aReadLaneValids(w) := fillIdx < chunkK
    }
  }
  val aWordsThisBeat = PopCount(aReadLaneValids)

  val prefetchReadDestIdxs = Wire(Vec(wordsPerBeat, UInt(chunkWidth.W)))
  val prefetchReadLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    if (wordsPerBeat > 1) {
      val laneAfterStart = w.U(laneWidth.W) >= prefetchALane
      val laneDelta = Mux(laneAfterStart, w.U(laneWidth.W) - prefetchALane, 0.U(laneWidth.W))
      prefetchReadDestIdxs(w) := prefetchFillIdx + laneDelta
      prefetchReadLaneValids(w) := laneAfterStart && (prefetchReadDestIdxs(w) < prefetchedChunkK)
    } else {
      prefetchReadDestIdxs(w) := prefetchFillIdx
      prefetchReadLaneValids(w) := prefetchFillIdx < prefetchedChunkK
    }
  }
  val prefetchWordsThisBeat = PopCount(prefetchReadLaneValids)

  val bRequestedK = bCacheK(cacheIdxWidth - 1, 0)
  val bFillBusyMask = bFillInflight.asUInt
  val bHasFreeSource = !bFillBusyMask.andR
  val bFillIssueSourceOH = PriorityEncoderOH(~bFillBusyMask)
  val bFillIssueSource = OHToUInt(bFillIssueSourceOH)
  val bIssueWords = Mux(
    (bRequestedK - bIssueIdx) > wordsPerBeat.U,
    wordsPerBeat.U(cacheIdxWidth.W),
    bRequestedK - bIssueIdx
  )
  val bIssueAddr = bBase + bIssueIdx * xlenBytes.U
  val getBCacheIssueBits = edgesOut.Get(
    fromSource = bFillIssueSource,
    toAddress = bIssueAddr,
    lgSize = readLgSize
  )._2
  val bRespSource = tlOut.d.bits.source(tlSourceIdxWidth - 1, 0)
  val bRespBaseIdx = bFillInflightIdx(bRespSource)
  val bRespDestIdxs = Wire(Vec(wordsPerBeat, UInt(cacheIdxWidth.W)))
  val bRespLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    bRespDestIdxs(w) := bRespBaseIdx + w.U
    bRespLaneValids(w) := bRespDestIdxs(w) < bRequestedK
  }
  val bRespWordsThisBeat = PopCount(bRespLaneValids)

  val writeLaneWordIdxs = Wire(Vec(wordsPerBeat, UInt(outIdxWidth.W)))
  val writeLaneValids = Wire(Vec(wordsPerBeat, Bool()))
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
  val writeWordsThisBeat = PopCount(writeLaneValids)

  val putDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  if (useBFloat16Output) {
    for (w <- 0 until wordsPerBeat) {
      putDataWords(w) := Mux(writeLaneValids(w), packedStoreWordsReg.get(writeLaneWordIdxs(w)), 0.U)
    }
  } else {
    for (w <- 0 until wordsPerBeat) {
      putDataWords(w) := Mux(writeLaneValids(w), flatOut(writeLaneWordIdxs(w)).pad(xLen), 0.U)
    }
  }
  val putBits = edgesOut.Put(
    fromSource = 0.U,
    toAddress = writeBeatAddr,
    lgSize = readLgSize,
    data = putDataWords.asUInt
  )._2

  val readDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    readDataWords(w) := tlOut.d.bits.data((w + 1) * xLen - 1, w * xLen)
  }

  tlOut.a.valid := false.B
  tlOut.a.bits := getABits
  tlOut.d.ready := false.B

  io.cmd.ready := (state === s_idle)
  io.resp.valid := (state === s_resp)
  io.resp.bits.rd := respRd
  io.resp.bits.data := respData

  val clearPerfCounters = WireDefault(false.B)
  val incRunCmd = WireDefault(false.B)
  val incPreloadCmd = WireDefault(false.B)
  val incPreloadReuseHit = WireDefault(false.B)
  val incChunkStarted = WireDefault(false.B)
  val incTLBRead = WireDefault(false.B)
  val incTLARead = WireDefault(false.B)
  val incTLCWrite = WireDefault(false.B)

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
      totalK := requestedK
      kRemaining := requestedK
      kBaseIdx := 0.U
      chunkK := 0.U
      fillIdx := 0.U
      feedRowIdx := 0.U
      captureRowIdx := 0.U
      outIdx := 0.U
      bCacheFillIdx := 0.U
      bIssueIdx := 0.U
      activeABufSel := 0.U
      prefetchedABufSel := 1.U
      prefetchedValid := false.B
      prefetchedBaseIdx := 0.U
      prefetchedChunkK := 0.U
      prefetchFillIdx := 0.U
      prefetchState := pf_idle
      for (i <- 0 until tlSourceIds) {
        bFillInflight(i) := false.B
      }
      for (r <- 0 until nRows) {
        for (c <- 0 until nCols) {
          accumTile(r)(c) := 0.U
        }
      }
      val canReusePinnedWeights =
        pinnedWeightsValid && bCacheValid && (bCacheK === requestedK) && (cachedBBase === bBase)
      when(canReusePinnedWeights) {
        incPreloadReuseHit := true.B
      }
      bFillForRun := !canReusePinnedWeights
      when(!canReusePinnedWeights) {
        bCacheValid := requestedK === 0.U
        pinnedWeightsValid := false.B
      }
      bCacheK := requestedK
      respData := Mux(!configured, 1.U, Mux(requestedK > maxK.U, 2.U, 0.U))
      state := Mux(
        !configured || requestedK > maxK.U,
        s_resp,
        Mux(canReusePinnedWeights || requestedK === 0.U, s_prep_chunk, s_req_fill_b_cache)
      )
    }.elsewhen(funct === 6.U) {
      incRunCmd := true.B
      val requestedK = bCacheK
      aBase := io.cmd.bits.rs1
      cBase := io.cmd.bits.rs2
      totalK := requestedK
      kRemaining := requestedK
      kBaseIdx := 0.U
      chunkK := 0.U
      fillIdx := 0.U
      feedRowIdx := 0.U
      captureRowIdx := 0.U
      outIdx := 0.U
      bCacheFillIdx := 0.U
      bIssueIdx := 0.U
      activeABufSel := 0.U
      prefetchedABufSel := 1.U
      prefetchedValid := false.B
      prefetchedBaseIdx := 0.U
      prefetchedChunkK := 0.U
      prefetchFillIdx := 0.U
      prefetchState := pf_idle
      for (i <- 0 until tlSourceIds) {
        bFillInflight(i) := false.B
      }
      when(pinnedWeightsValid) {
        incPreloadReuseHit := true.B
      }
      bFillForRun := false.B
      for (r <- 0 until nRows) {
        for (c <- 0 until nCols) {
          accumTile(r)(c) := 0.U
        }
      }
      respData := Mux(!pinnedWeightsValid, 3.U, Mux(requestedK > maxK.U, 2.U, 0.U))
      state := Mux(
        !pinnedWeightsValid || requestedK > maxK.U,
        s_resp,
        s_prep_chunk
      )
    }.elsewhen(funct === 2.U) {
      respData := Cat(0.U((xLen - 2).W), state =/= s_idle, state === s_resp)
      state := s_resp
    }.elsewhen(funct === 3.U) {
      incPreloadCmd := true.B
      val requestedK = io.cmd.bits.rs2
      bBase := io.cmd.bits.rs1
      bCacheFillIdx := 0.U
      bIssueIdx := 0.U
      prefetchedValid := false.B
      prefetchFillIdx := 0.U
      prefetchState := pf_idle
      for (i <- 0 until tlSourceIds) {
        bFillInflight(i) := false.B
      }
      bCacheK := requestedK
      bCacheValid := requestedK === 0.U
      pinnedWeightsValid := requestedK === 0.U
      cachedBBase := io.cmd.bits.rs1
      bFillForRun := false.B
      respData := Mux(requestedK > maxK.U, 2.U, 0.U)
      state := Mux(
        requestedK > maxK.U || requestedK === 0.U,
        s_resp,
        s_req_fill_b_cache
      )
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

  when(core.io.outValid) {
    for (c <- 0 until nCols) {
      accumTile(captureRowIdx)(c) := accumTile(captureRowIdx)(c) + core.io.outVec(c)
    }
    captureRowIdx := captureRowIdx + 1.U
  }

  val remainingAfterCurrent = kRemaining - chunkK
  val canStartAPrefetch =
    (state === s_load_weights || state === s_feed_rows || state === s_wait_chunk_out) &&
      (remainingAfterCurrent =/= 0.U) &&
      !prefetchedValid &&
      (prefetchState === pf_idle)

  when(canStartAPrefetch) {
    prefetchedABufSel := activeABufSel ^ 1.U
    prefetchedBaseIdx := kBaseIdx + chunkK
    prefetchedChunkK := Mux(remainingAfterCurrent > chunkKMax.U, chunkKMax.U, remainingAfterCurrent)(chunkWidth - 1, 0)
    prefetchFillIdx := 0.U
    prefetchState := pf_req_fill_a
  }

  val bFillCanIssue =
    (state === s_req_fill_b_cache || state === s_wait_fill_b_cache) &&
      bHasFreeSource &&
      (bIssueIdx < bRequestedK)

  when(state === s_req_fill_b_cache) {
    tlOut.a.valid := bFillCanIssue
    tlOut.a.bits := getBCacheIssueBits
    when(tlOut.a.fire) {
      incTLBRead := true.B
      bFillInflight(bFillIssueSource) := true.B
      bFillInflightIdx(bFillIssueSource) := bIssueIdx
      bIssueIdx := bIssueIdx + bIssueWords
      state := s_wait_fill_b_cache
    }
  }

  when(state === s_wait_fill_b_cache) {
    tlOut.a.valid := bFillCanIssue
    tlOut.a.bits := getBCacheIssueBits
    tlOut.d.ready := true.B
    when(tlOut.a.fire) {
      incTLBRead := true.B
      bFillInflight(bFillIssueSource) := true.B
      bFillInflightIdx(bFillIssueSource) := bIssueIdx
      bIssueIdx := bIssueIdx + bIssueWords
    }
    when(tlOut.d.fire) {
      for (w <- 0 until wordsPerBeat) {
        when(bRespLaneValids(w)) {
          bCacheBuf(bRespDestIdxs(w)) := readDataWords(w)
        }
      }
      bFillInflight(bRespSource) := false.B
      val nextIdx = bCacheFillIdx + bRespWordsThisBeat
      when(nextIdx >= bRequestedK) {
        bCacheFillIdx := 0.U
        bCacheValid := true.B
        when(!bFillForRun) {
          pinnedWeightsValid := true.B
          cachedBBase := bBase
        }
        state := Mux(bFillForRun, s_prep_chunk, s_resp)
      }.otherwise {
        bCacheFillIdx := nextIdx
      }
    }
  }

  when(state === s_prep_chunk) {
    when(kRemaining === 0.U) {
      outIdx := 0.U
      if (useBFloat16Output) {
        state := s_quantize_out
      } else {
        state := s_req_put
      }
    }.otherwise {
      when(prefetchedValid && (prefetchedBaseIdx === kBaseIdx)) {
        incChunkStarted := true.B
        activeABufSel := prefetchedABufSel
        chunkK := prefetchedChunkK
        prefetchedValid := false.B
        state := s_load_weights
      }.otherwise {
        incChunkStarted := true.B
        val nextChunk = Mux(kRemaining > chunkKMax.U, chunkKMax.U, kRemaining)
        chunkK := nextChunk(chunkWidth - 1, 0)
        fillIdx := 0.U
        state := s_req_fill_a
      }
    }
  }

  when(state === s_req_fill_a) {
    tlOut.a.valid := true.B
    tlOut.a.bits := getABits
    when(tlOut.a.fire) {
      incTLARead := true.B
      state := s_wait_fill_a
    }
  }

  when(state === s_wait_fill_a) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      for (w <- 0 until wordsPerBeat) {
        when(aReadLaneValids(w)) {
          aChunkBufs(activeABufSel)(aReadDestIdxs(w)) := readDataWords(w)
        }
      }
      val nextIdx = fillIdx + aWordsThisBeat
      when(nextIdx >= chunkK) {
        fillIdx := 0.U
        state := s_load_weights
      }.otherwise {
        fillIdx := nextIdx
        state := s_req_fill_a
      }
    }
  }

  when(prefetchState === pf_req_fill_a &&
      (state === s_load_weights || state === s_feed_rows || state === s_wait_chunk_out)) {
    tlOut.a.valid := true.B
    tlOut.a.bits := getPrefetchABits
    when(tlOut.a.fire) {
      incTLARead := true.B
      prefetchState := pf_wait_fill_a
    }
  }

  when(prefetchState === pf_wait_fill_a &&
      (state === s_load_weights || state === s_feed_rows || state === s_wait_chunk_out)) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      for (w <- 0 until wordsPerBeat) {
        when(prefetchReadLaneValids(w)) {
          aChunkBufs(prefetchedABufSel)(prefetchReadDestIdxs(w)) := readDataWords(w)
        }
      }
      val nextPrefetchIdx = prefetchFillIdx + prefetchWordsThisBeat
      when(nextPrefetchIdx >= prefetchedChunkK) {
        prefetchFillIdx := 0.U
        prefetchedValid := true.B
        prefetchState := pf_idle
      }.otherwise {
        prefetchFillIdx := nextPrefetchIdx
        prefetchState := pf_req_fill_a
      }
    }
  }

  when(state === s_load_weights) {
    core.io.loadWeights := true.B
    feedRowIdx := 0.U
    captureRowIdx := 0.U
    state := s_feed_rows
  }

  when(state === s_feed_rows) {
    core.io.inValid := true.B
    when(feedRowIdx === (nRows - 1).U) {
      state := s_wait_chunk_out
    }
    feedRowIdx := feedRowIdx + 1.U
  }

  when(state === s_wait_chunk_out) {
    when(captureRowIdx === nRows.U) {
      val remAfter = kRemaining - chunkK
      when(remAfter === 0.U || prefetchedValid) {
        kRemaining := remAfter
        kBaseIdx := kBaseIdx + chunkK
        state := s_prep_chunk
      }
    }
  }

  if (useBFloat16Output) {
    when(state === s_quantize_out) {
      for (w <- 0 until outputWordCount) {
        packedStoreWordsReg.get(w) := packedStoreWordsWire.get(w)
      }
      state := s_req_put
    }
  }

  when(state === s_req_put) {
    tlOut.a.valid := true.B
    tlOut.a.bits := putBits
    when(tlOut.a.fire) {
      incTLCWrite := true.B
      state := s_wait_put
    }
  }

  when(state === s_wait_put) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
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
    when(core.io.inValid) {
      perfCounters(perfFeedCycles) := perfCounters(perfFeedCycles) + 1.U
    }
    when(core.io.outValid) {
      perfCounters(perfCaptureRows) := perfCounters(perfCaptureRows) + 1.U
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
    when(state === s_wait_fill_b_cache) {
      perfCounters(perfWaitFillBCycles) := perfCounters(perfWaitFillBCycles) + 1.U
    }
    when(state === s_wait_fill_a || prefetchState === pf_wait_fill_a) {
      perfCounters(perfWaitFillACycles) := perfCounters(perfWaitFillACycles) + 1.U
    }
    when(state === s_load_weights) {
      perfCounters(perfLoadWeightCycles) := perfCounters(perfLoadWeightCycles) + 1.U
    }
    when(state === s_wait_chunk_out) {
      perfCounters(perfWaitChunkOutCycles) := perfCounters(perfWaitChunkOutCycles) + 1.U
    }
    when(state === s_wait_put) {
      perfCounters(perfWaitPutCycles) := perfCounters(perfWaitPutCycles) + 1.U
    }
  }
}

class SystolicArrayWSRoCC(
  opcodes: OpcodeSet,
  val precision: Int = 16,
  val nRows: Int = 4,
  val nCols: Int = 4,
  val maxK: Int = 64,
  val useBFloat16Input: Boolean = false,
  val useBFloat16Output: Boolean = false,
  val fixedPointFracBits: Int = 8,
  val accumBits: Int = 32,
  val numTLSourceIds: Int = 4
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new SystolicArrayWSImpl(this)
  override val atlNode = TLClientNode(
    Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1(
      name = "SystolicArrayWSRoCC",
      sourceId = IdRange(0, numTLSourceIds)
    ))))
  )
}
