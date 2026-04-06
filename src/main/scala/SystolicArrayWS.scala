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

  // Bit-width of each input element from A/B.
  // Packing means combining several precision-bit elements into one xLen word.
  // Slicing means extracting one precision-bit lane back out of that packed word.
  private val precision = outer.precision // 16bit
  // Number of rows in the PE mesh.
  // Also sets the height of one A tile and the number of A lanes fed per cycle.
  private val nRows = outer.nRows // 4
  // Number of columns in the PE mesh.
  // Also sets the width of one B tile and the width of one output tile.
  private val nCols = outer.nCols // 4
  // Maximum K supported by one command.
  // Software/hardware may still process that K in smaller chunks.
  // Increasing maxK mostly increases capacity, not per-cycle throughput.
  // It can avoid extra command-level tiling for larger problems, but also grows storage.
  // bCacheBuf is the on-accelerator buffer that stores the packed B stream for up to maxK words.
  // maxK is the total command-level K capacity; chunkKMax is the per-chunk K consumed by the mesh.
  private val maxK = outer.maxK // 64
  // Selects the BF16-input datapath, which converts BF16 inputs to fixed-point before MAC.
  private val useBFloat16Input = outer.useBFloat16Input // True
  // Selects BF16 output packing.
  // When true, accumulated outputs are converted back to BF16 and several outputs can share one xLen word.
  private val useBFloat16Output = outer.useBFloat16Output // True
  // Fractional-bit count used by the fixed-point representation in the BF16-input path.
  private val fixedPointFracBits = outer.fixedPointFracBits // 12
  // Maximum K processed by one mesh load/feed/capture chunk.
  // This is tied to array shape: one chunk uses at most nRows weight rows.
  // maxK sizes the whole run; chunkKMax sizes one step of that run.
  private val chunkKMax = nRows // 4
  // Bit-width needed to count values in the range [0, maxK].
  private val kWidth = log2Ceil(maxK + 1) // 7
  // Bit-width needed to index bCacheBuf, the local packed-B storage.
  private val cacheIdxWidth = log2Ceil(maxK + 1) // 7
  // Bit-width for indices local to one chunk.
  // fillIdx tracks how many A words of the current chunk are loaded.
  // feedRowIdx tracks which chunk row is currently being fed into the core.
  // captureRowIdx tracks which output row from the chunk is being accumulated.
  // Since those counters never exceed chunkKMax/nRows, chunkWidth can be smaller than kWidth.
  private val chunkWidth = log2Ceil(chunkKMax + 1) // 3
  // Bit-width of each accumulated output element inside accumTile.
  private val accumPrec = outer.accumBits // 64
  // Number of scalar output elements in one output tile.
  private val outCount = nRows * nCols // 16
  // Number of output elements that fit into one architectural word during writeback.
  // With BF16 output on RV64, this is 64 / 16 = 4 elements per word.
  private val outputElemsPerWord = if (useBFloat16Output) (xLen / precision) else 1 // 4
  // Number of architectural words needed to store the whole output tile.
  // Example: 16 outputs / 4 BF16 outputs per word = 4 output words.
  // Used to size the output-word index and packedStoreWords* buffers.
  private val outputWordCount = (outCount + outputElemsPerWord - 1) / outputElemsPerWord // 4
  // Bit-width for indexing output words during writeback.
  private val outIdxWidth = log2Ceil(outputWordCount + 1) // 3
  // Architectural word size in bytes.
  private val xlenBytes = xLen / 8 // 8 bytes
  // log2(xlenBytes).
  // Example: RV64 => xlenBytes = 8, so lgXlenBytes = 3.
  // Used when extracting the word-lane offset from a byte address inside a TileLink beat.
  private val lgXlenBytes = log2Ceil(xlenBytes) // 3 bits
  // Number of bytes in one TileLink data beat.
  // Example: 128-bit bus => 16 bytes per beat.
  private val beatBytes = cacheDataBits / 8 // 16 bytes per beat
  // Number of architectural words contained in one TileLink beat.
  // Example: 16-byte beat / 8-byte xLen word = 2 words per beat.
  private val wordsPerBeat = beatBytes / xlenBytes // 2 words per beat
  // Number of TileLink source IDs the accelerator may use for outstanding requests.
  // Different source IDs let multiple requests stay in flight at once so responses can be matched back later.
  private val tlSourceIds = outer.numTLSourceIds
  // Bit-width needed to encode those TileLink source IDs.
  private val tlSourceIdxWidth = math.max(1, log2Ceil(tlSourceIds))
  // Bit-width needed to choose a word lane within one beat.
  // Example: 2 words/beat => laneWidth = 1 bit.
  private val laneWidth = math.max(1, log2Ceil(wordsPerBeat))
  // Number of low address bits needed to address bytes inside one beat.
  // Example: 16-byte beat => beatOffsetBits = 4, so address bits [3:0] are the within-beat byte offset.
  private val beatOffsetBits = log2Ceil(beatBytes)
  // These are private because they are internal derived constants used only by this module's control/datapath logic.

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

  // Select the datapath implementation.
  // BF16-input mode converts BF16 values to fixed-point before the mesh.
  // Otherwise the mesh consumes plain integer/fixed-point lanes directly.
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

  // Two local A chunk buffers used for ping-pong buffering, while one executes the other reads.
  // One buffer can feed the mesh while the other buffer is being prefetched from memory.
  // Each buffer stores up to chunkKMax packed A words, one xLen word per k-step in the chunk.
  val aChunkBufs = Reg(Vec(2, Vec(chunkKMax, UInt(xLen.W))))
  // Local packed-B storage for the whole command, up to maxK packed words.
  // This is the "B cache" the accelerator fills from memory before chunked execution begins.
  // Where we store the weights before execution
  val bCacheBuf = Reg(Vec(maxK, UInt(xLen.W)))
  // Accumulated output tile.
  // Each chunk contributes partial sums into this tile until the full K dimension is finished.
  val accumTile = RegInit(VecInit(Seq.fill(nRows)(VecInit(Seq.fill(nCols)(0.U(accumPrec.W))))))

  // Current nRows x nCols weight slice extracted from bCacheBuf for the active chunk.
  val weightTile = Wire(Vec(nRows, Vec(nCols, UInt(precision.W))))
  // Current vector presented to the core input on this feed cycle, one lane per row.
  val currentInVec = Wire(Vec(nRows, UInt(precision.W)))

  // Unpacked view of the active A chunk buffer. (Flatten the buffer, for convenience)
  // aChunkLanes(k)(r) = row-lane r extracted from packed A word k of the active buffer.
  val aChunkLanes = Wire(Vec(chunkKMax, Vec(nRows, UInt(precision.W))))

  // Flatten accumTile into row-major order so writeback logic can treat outputs as a linear vector.
  val flatOut = Wire(Vec(outCount, UInt(accumPrec.W)))
  for (r <- 0 until nRows) {
    for (c <- 0 until nCols) {
      flatOut(r * nCols + c) := accumTile(r)(c)
    }
  }

  // One BF16-converted output element per logical output position.
  val packedBFloatOut = if (useBFloat16Output) Some(Wire(Vec(outCount, UInt(precision.W)))) else None
  // Combinationally packed writeback words.
  // These group several BF16 outputs into one xLen word before they are optionally registered.
  val packedStoreWordsWire = if (useBFloat16Output) Some(Wire(Vec(outputWordCount, UInt(xLen.W)))) else None
  // Registered copy of the packed writeback words.
  // This decouples BF16 conversion/packing from the later TileLink put state.
  val packedStoreWordsReg = if (useBFloat16Output) Some(Reg(Vec(outputWordCount, UInt(xLen.W)))) else None

  if (useBFloat16Output) {

    // Convert each accumulated fixed-point result into one BF16 element.
    for (r <- 0 until nRows) {
      for (c <- 0 until nCols) {
        val conv = Module(new SIntFixedToBFloat16(accumPrec, 2 * fixedPointFracBits))
        conv.io.in := accumTile(r)(c).asSInt
        packedBFloatOut.get(r * nCols + c) := conv.io.out
      }
    }

    // Pack several BF16 output elements into one xLen-sized store word.
    // Any unused lanes in the last word are zero-padded.
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

  // Main control FSM for the RoCC wrapper:
  // fill B cache, prepare/fill A chunks(inputs), load weights, feed the core, optionally quantize outputs, write results, respond.
  val s_idle :: s_req_fill_b_cache :: s_wait_fill_b_cache :: s_prep_chunk :: s_req_fill_a :: s_wait_fill_a :: s_load_weights :: s_feed_rows :: s_wait_chunk_out :: s_quantize_out :: s_req_put :: s_wait_put :: s_resp :: Nil = Enum(13)
  // Current FSM state.
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

  // Base address of the packed A stream in memory.
  val aBase = RegInit(0.U(xLen.W))
  // Base address of the packed B stream in memory.
  val bBase = RegInit(0.U(xLen.W))
  // Base address of the output tile buffer in memory.
  val cBase = RegInit(0.U(xLen.W))
  // Set after a valid configure/preload setup so later run commands know the accelerator is initialized.
  val configured = RegInit(false.B)

  // Total K requested by the current run command. (M, N, K)
  val totalK = RegInit(0.U(xLen.W))
  // Remaining K still to be processed for the current run.
  val kRemaining = RegInit(0.U(xLen.W))
  // Starting K index of the current chunk inside the full run. (after 1 run may become 4)
  val kBaseIdx = RegInit(0.U(xLen.W))
  // K size of the current chunk, capped by chunkKMax. ( can be 1, 2, ... 4 )
  val chunkK = RegInit(0.U(chunkWidth.W))
  // Next A-word slot to fill in the active chunk buffer.
  val fillIdx = RegInit(0.U(chunkWidth.W))
  // Which feed cycle / chunk row is currently being sent into the core.
  val feedRowIdx = RegInit(0.U(chunkWidth.W))
  // Which output row from the current chunk is being captured into accumTile. (which one is ready)
  val captureRowIdx = RegInit(0.U(chunkWidth.W))
  // Output-word writeback index.
  // Indexes output words, not scalar output elements.
  val outIdx = RegInit(0.U(outIdxWidth.W))
  // Next bCacheBuf slot to be committed by the serialized B-fill path.
  val bCacheFillIdx = RegInit(0.U(cacheIdxWidth.W))
  // Next packed-B word index to issue as a TileLink request.
  val bIssueIdx = RegInit(0.U(cacheIdxWidth.W))
  // True when bCacheBuf holds the requested B stream contents.
  val bCacheValid = RegInit(false.B)
  // True when the currently cached B weights may be reused by later run-preloaded commands.
  val pinnedWeightsValid = RegInit(false.B)
  // K length associated with the data currently expected/stored in bCacheBuf.
  val bCacheK = RegInit(0.U(xLen.W))
  // B base address associated with the cached/pinned weights.
  val cachedBBase = RegInit(0.U(xLen.W))
  // True when the current B fill is happening only for this run and should not become pinned reusable weights.
  val bFillForRun = RegInit(false.B)
  // One bit per TileLink source ID indicating whether a B-fill request is still in flight on that source.
  val bFillInflight = RegInit(VecInit(Seq.fill(tlSourceIds)(false.B)))
  // Remembers which bCacheBuf index each in-flight B-fill request will write back into when its response returns.
  val bFillInflightIdx = Reg(Vec(tlSourceIds, UInt(cacheIdxWidth.W)))

  // Destination integer register for the RoCC response.
  val respRd = RegInit(0.U(5.W))
  // Data payload returned on the RoCC response channel.
  val respData = RegInit(0.U(xLen.W))

  // Which ping-pong A buffer is currently feeding the core.
  val activeABufSel = RegInit(0.U(1.W))
  // Which ping-pong A buffer is currently reserved for prefetch.
  val prefetchedABufSel = RegInit(1.U(1.W))
  // True when the prefetched A buffer is ready to become active.
  val prefetchedValid = RegInit(false.B)
  // Starting K index of the prefetched chunk.
  val prefetchedBaseIdx = RegInit(0.U(xLen.W))
  // K size of the prefetched chunk.
  val prefetchedChunkK = RegInit(0.U(chunkWidth.W))
  // Fill index used by the A-prefetch path.
  val prefetchFillIdx = RegInit(0.U(chunkWidth.W))

  // Mini-FSM that prefetches the next A chunk in parallel with current compute.
  val pf_idle :: pf_req_fill_a :: pf_wait_fill_a :: Nil = Enum(3)
  // Current A-prefetch FSM state.
  val prefetchState = RegInit(pf_idle)

  // Extract the active nRows x nCols weight tile from the packed B cache.
  // Each bCacheBuf word holds one K-step worth of nCols weight lanes.
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

  // Unpack the active A chunk buffer into lane form so the current feed cycle can select one vector cleanly.
  for (k <- 0 until chunkKMax) {
    for (r <- 0 until nRows) {
      aChunkLanes(k)(r) := aChunkBufs(activeABufSel)(k)(precision * (r + 1) - 1, precision * r)
    }
  }

  // Clamp feedRowIdx when the counter has already advanced past the valid mesh rows.
  val safeFeedRowIdx = WireDefault(0.U(chunkWidth.W))
  when(feedRowIdx < nRows.U) {
    safeFeedRowIdx := feedRowIdx
  }

  // Build the per-cycle core input vector from the active chunk.
  // currentInVec(i) picks row-lane i from the packed A word selected by safeFeedRowIdx.
  // Lanes beyond the current chunk size are zero-padded.
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

  // TileLink client port used for A/B reads and C writes.
  val (tlOut, edgesOut) = outer.atlNode.out(0)

  // Converting local (tile/chunk) indices to global ones (memory).
  // Absolute A-word index currently being filled for the active chunk.
  val absFillIdx = kBaseIdx + fillIdx
  // Byte address of the current A word to fetch.
  val aAddr = aBase + absFillIdx * xlenBytes.U
  // Byte address of the current output word to write back.
  val writeAddr = cBase + outIdx * xlenBytes.U
  // Word-lane of aAddr inside its TileLink beat.
  val aLane = if (wordsPerBeat > 1) aAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  // Word-lane of writeAddr inside its TileLink beat.
  val writeLane = if (wordsPerBeat > 1) writeAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)

  // Beat-aligned word index for the A fetch.
  val aBeatBaseIdx = if (wordsPerBeat > 1) (absFillIdx >> laneWidth) << laneWidth else absFillIdx
  // Beat-aligned word index for the C writeback.
  val writeBeatBaseIdx = if (wordsPerBeat > 1) (outIdx >> laneWidth) << laneWidth else outIdx

  // Beat-aligned byte address for the A fetch.
  val aBeatAddr = aBase + aBeatBaseIdx * xlenBytes.U
  // Beat-aligned byte address for the C writeback.
  val writeBeatAddr = cBase + writeBeatBaseIdx * xlenBytes.U

  // TileLink transfer size:
  // one full beat when multiple xLen words fit per beat, otherwise one xLen word.
  val readLgSize = if (wordsPerBeat > 1) beatOffsetBits.U else lgXlenBytes.U

  // Prebuilt TileLink Get message for active A-chunk fills.
  val getABits = edgesOut.Get(fromSource = 0.U, toAddress = aBeatAddr, lgSize = readLgSize)._2

  // Address generation for the A-prefetch path, parallel to the active A-fill path.
  val prefetchAAddr = aBase + (prefetchedBaseIdx + prefetchFillIdx) * xlenBytes.U
  val prefetchALane = if (wordsPerBeat > 1) prefetchAAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  val prefetchABeatBaseIdx = if (wordsPerBeat > 1) ((prefetchedBaseIdx + prefetchFillIdx) >> laneWidth) << laneWidth else (prefetchedBaseIdx + prefetchFillIdx)
  val prefetchABeatAddr = aBase + prefetchABeatBaseIdx * xlenBytes.U
  // Prebuilt TileLink Get message for prefetched A chunks.
  val getPrefetchABits = edgesOut.Get(fromSource = 0.U, toAddress = prefetchABeatAddr, lgSize = readLgSize)._2

  // For each word lane returned in a TileLink beat, compute which A-chunk slot it should fill.
  val aReadDestIdxs = Wire(Vec(wordsPerBeat, UInt(chunkWidth.W)))
  // Valid mask for those returned lanes: only lanes at/after aLane and still inside chunkK are useful.
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

  // Number of packed B words needed for the current preload/run.
  val bRequestedK = bCacheK(cacheIdxWidth - 1, 0)
  // Bitmask of TileLink source IDs currently occupied by in-flight B reads.
  val bFillBusyMask = bFillInflight.asUInt
  // True when at least one TileLink source ID is free to issue another B read.
  val bHasFreeSource = !bFillBusyMask.andR
  // One-hot selection of the next free TileLink source ID.
  val bFillIssueSourceOH = PriorityEncoderOH(~bFillBusyMask)
  // Integer index of the chosen TileLink source ID.
  val bFillIssueSource = OHToUInt(bFillIssueSourceOH)
  // Number of useful packed B words expected from the next issued beat.
  val bIssueWords = Mux(
    (bRequestedK - bIssueIdx) > wordsPerBeat.U,
    wordsPerBeat.U(cacheIdxWidth.W),
    bRequestedK - bIssueIdx
  )
  // Byte address of the next packed B read request.
  val bIssueAddr = bBase + bIssueIdx * xlenBytes.U
  // Prebuilt TileLink Get message for B-cache fill traffic.
  val getBCacheIssueBits = edgesOut.Get(
    fromSource = bFillIssueSource,
    toAddress = bIssueAddr,
    lgSize = readLgSize
  )._2
  // Source ID carried by the returning B-fill response.
  val bRespSource = tlOut.d.bits.source(tlSourceIdxWidth - 1, 0)
  // Starting bCacheBuf index associated with that response's original request.
  val bRespBaseIdx = bFillInflightIdx(bRespSource)
  // Destination bCacheBuf slot for each returned word lane in the response beat.
  val bRespDestIdxs = Wire(Vec(wordsPerBeat, UInt(cacheIdxWidth.W)))
  // Valid mask for returned B lanes; last beat may be only partially useful.
  val bRespLaneValids = Wire(Vec(wordsPerBeat, Bool()))
  for (w <- 0 until wordsPerBeat) {
    bRespDestIdxs(w) := bRespBaseIdx + w.U
    bRespLaneValids(w) := bRespDestIdxs(w) < bRequestedK
  }
  // Number of useful packed B words carried by this response beat.
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
