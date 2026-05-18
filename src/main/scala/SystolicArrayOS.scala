package girdap
import chisel3._
import chisel3.util._

import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink.{TLClientNode, TLMasterParameters, TLMasterPortParameters}
import org.chipsalliance.cde.config.Parameters

// Ouptut Stationary Processing Element
class ProcessingElement(inPrec: Int = 16, accumPrec: Int = 32) extends Module {
  val io = IO(new Bundle {
    // Signals
    val clear = Input(Bool())
    val inValid = Input(Bool())
    val outValid = Output(Bool())

    // InData
    val left = Input(UInt(inPrec.W))
    val top  = Input(UInt(inPrec.W))

    // OutData
    val right  = Output(UInt(inPrec.W))
    val down   = Output(UInt(inPrec.W))
    val result = Output(UInt(accumPrec.W))
  })

  val accum = RegInit(0.U(accumPrec.W))
  val down  = RegInit(0.U(inPrec.W))
  val right = RegInit(0.U(inPrec.W))
  val v = RegInit(false.B)

  val mul = io.top * io.left
  val nextAccum = mul + accum

  when(io.clear){
    accum := 0.U
    v := false.B
  }.otherwise{
    v := io.inValid
    when(io.inValid) {
      accum := nextAccum
    }
  }

  down := io.top
  right := io.left

  io.right := right
  io.down := down
  io.result := accum

  io.outValid := v
}

// Output Stationary Mesh
class Mesh(precision: Int = 16, nRows: Int = 2, nCols: Int = 2) extends Module {
  val io = IO(new Bundle {
    val left = Input(Vec(nRows, UInt(precision.W)))
    val top = Input(Vec(nCols, UInt(precision.W)))
    val inValid = Input(Bool())
    val clear = Input(Bool())

    val out = Output(Vec(nRows, Vec(nCols, UInt((2*precision).W))))
  })
  val PEs = Seq.fill(nRows)(
    Seq.fill(nCols)(Module(new ProcessingElement(inPrec = precision, accumPrec = precision * 2)))
  )

  // Corner
  PEs(0)(0).io.left := io.left(0)
  PEs(0)(0).io.top := io.top(0)
  PEs(0)(0).io.inValid := io.inValid

  // Left Edge
  for(i <- 1 until nRows){
    PEs(i)(0).io.left := io.left(i)
    PEs(i)(0).io.top := PEs(i-1)(0).io.down
    PEs(i)(0).io.inValid := PEs(i-1)(0).io.outValid
  }

  // Top Edge
  for(j <- 1 until nCols){
    PEs(0)(j).io.top := io.top(j)
    PEs(0)(j).io.left := PEs(0)(j-1).io.right
    PEs(0)(j).io.inValid := PEs(0)(j-1).io.outValid
  }

  // Only inner ones
  for(i <- 1 until nRows){
    for(j <- 1 until nCols){
      PEs(i)(j).io.top := PEs(i-1)(j).io.down
      PEs(i)(j).io.left := PEs(i)(j-1).io.right
      PEs(i)(j).io.inValid := PEs(i)(j-1).io.outValid && PEs(i-1)(j).io.outValid
    }
  }

  // All the PEs
  for(i <- 0 until nRows) {
    for (j <- 0 until nCols) {
      PEs(i)(j).io.clear := io.clear
      io.out(i)(j) := PEs(i)(j).io.result
    }
  }
}

class Controller(
  precision: Int = 16,
  nRows: Int = 2,
  nCols: Int = 2,
  maxK: Int = 64
) extends Module {
  require(nRows > 0, "nRows must be > 0")
  require(nCols > 0, "nCols must be > 0")
  require(maxK > 0, "maxK must be > 0")

  private val kWidth = log2Ceil(maxK + 1)
  private val tWidth = log2Ceil(maxK + nRows + nCols + 1)
  private val queueDepth = maxK + math.max(nRows, nCols)

  val io = IO(new Bundle {
    // Command/control.
    val cmdValid = Input(Bool())
    val cmdReady = Output(Bool())
    val k = Input(UInt(kWidth.W))
    val clearOnStart = Input(Bool())

    // Load one k-slice per cycle while loading.
    val inValid = Input(Bool())
    val inReady = Output(Bool())
    val in_left = Input(Vec(nRows, UInt(precision.W)))
    val in_top = Input(Vec(nCols, UInt(precision.W)))

    // Streamed outputs to mesh edges.
    val out_left = Output(Vec(nRows, UInt(precision.W)))
    val out_top = Output(Vec(nCols, UInt(precision.W)))
    val outValid = Output(Bool())
    val clear = Output(Bool())

    // Status.
    val busy = Output(Bool())
    val done = Output(Bool())
  })

  val leftQ = Seq.fill(nRows)(Module(new Queue(UInt(precision.W), queueDepth)))
  val topQ = Seq.fill(nCols)(Module(new Queue(UInt(precision.W), queueDepth)))

  val sIdle :: sLoad :: sClear :: sStream :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)

  val kReg = RegInit(0.U(kWidth.W))
  val loadCount = RegInit(0.U(kWidth.W))
  val tCount = RegInit(0.U(tWidth.W))
  val clearThisRun = RegInit(true.B)

  val totalCycles = Wire(UInt(tWidth.W))
  totalCycles := kReg + (nRows + nCols - 2).U

  val allLeftEnqReady = leftQ.map(_.io.enq.ready).reduce(_ && _)
  val allTopEnqReady = topQ.map(_.io.enq.ready).reduce(_ && _)
  val canLoad = allLeftEnqReady && allTopEnqReady
  val doLoad = (state === sLoad) && io.inValid && canLoad

  io.cmdReady := (state === sIdle)
  io.inReady := (state === sLoad) && canLoad
  io.clear := (state === sClear)
  io.outValid := (state === sStream) && (tCount < kReg)
  io.busy := (state =/= sIdle)
  io.done := (state === sDone)

  for (i <- 0 until nRows) {
    leftQ(i).io.enq.valid := doLoad
    leftQ(i).io.enq.bits := io.in_left(i)

    leftQ(i).io.deq.ready := false.B
    io.out_left(i) := 0.U
  }

  for (j <- 0 until nCols) {
    topQ(j).io.enq.valid := doLoad
    topQ(j).io.enq.bits := io.in_top(j)

    topQ(j).io.deq.ready := false.B
    io.out_top(j) := 0.U
  }

  switch(state) {
    is(sIdle) {
      when(io.cmdValid) {
        kReg := io.k
        loadCount := 0.U
        tCount := 0.U
        clearThisRun := io.clearOnStart
        when(io.k === 0.U) {
          state := Mux(io.clearOnStart, sClear, sDone)
        }.otherwise {
          state := sLoad
        }
      }
    }

    is(sLoad) {
      when(doLoad) {
        val nextLoadCount = loadCount + 1.U
        loadCount := nextLoadCount
        when(nextLoadCount === kReg) {
          state := Mux(clearThisRun, sClear, sStream)
        }
      }
    }

    is(sClear) {
      tCount := 0.U
      state := Mux(kReg === 0.U, sDone, sStream)
    }

    is(sStream) {
      for (i <- 0 until nRows) {
        val active = (tCount >= i.U) && (tCount < (kReg + i.U))
        leftQ(i).io.deq.ready := active
        io.out_left(i) := Mux(active && leftQ(i).io.deq.valid, leftQ(i).io.deq.bits, 0.U)
      }
      for (j <- 0 until nCols) {
        val active = (tCount >= j.U) && (tCount < (kReg + j.U))
        topQ(j).io.deq.ready := active
        io.out_top(j) := Mux(active && topQ(j).io.deq.valid, topQ(j).io.deq.bits, 0.U)
      }

      when((tCount + 1.U) >= totalCycles) {
        state := sDone
      }.otherwise {
        tCount := tCount + 1.U
      }
    }

    is(sDone) {
      state := sIdle
    }
  }
}

class SystolicArrayCore(
  precision: Int = 16,
  nRows: Int = 2,
  nCols: Int = 2,
  maxK: Int = 64
) extends Module {
  private val kWidth = log2Ceil(maxK + 1)

  val io = IO(new Bundle {
    // Command/control for one tile run.
    val cmdValid = Input(Bool())
    val cmdReady = Output(Bool())
    val k = Input(UInt(kWidth.W))
    val cmdClear = Input(Bool())

    // K-slice load interface.
    val inValid = Input(Bool())
    val inReady = Output(Bool())
    val in_left = Input(Vec(nRows, UInt(precision.W)))
    val in_top = Input(Vec(nCols, UInt(precision.W)))

    // Tile outputs and status.
    val out = Output(Vec(nRows, Vec(nCols, UInt((2 * precision).W))))
    val busy = Output(Bool())
    val done = Output(Bool())
  })

  val controller = Module(new Controller(precision = precision, nRows = nRows, nCols = nCols, maxK = maxK))
  val mesh = Module(new Mesh(precision = precision, nRows = nRows, nCols = nCols))

  controller.io.cmdValid := io.cmdValid
  io.cmdReady := controller.io.cmdReady
  controller.io.k := io.k
  controller.io.clearOnStart := io.cmdClear

  controller.io.inValid := io.inValid
  io.inReady := controller.io.inReady
  controller.io.in_left := io.in_left
  controller.io.in_top := io.in_top

  mesh.io.left := controller.io.out_left
  mesh.io.top := controller.io.out_top
  mesh.io.inValid := controller.io.outValid
  mesh.io.clear := controller.io.clear

  io.out := mesh.io.out
  io.busy := controller.io.busy
  io.done := controller.io.done
}

class SystolicArrayImpl(
  outer: SystolicArrayOSRoCC
)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters
  with HasL1CacheParameters {

  val cacheParams = tileParams.dcache.get

  // Input Precision ( 16 Bits )
  private val precision = outer.precision
  // Mesh Size ( 2x2 )
  private val nRows = outer.nRows
  private val nCols = outer.nCols
  // Common Dimension ( MxK @ KxN, 2x8 @ 8x2 )
  private val maxK = outer.maxK
  // Common Dimension Size ( 4 Bits )
  private val kWidth = log2Ceil(maxK + 1)
  // Result precision ( 32 Bits )
  private val accumPrec = precision * 2
  // Output Size ( 2x8 @ 8x2 = 2x2 )
  private val outCount = nRows * nCols
  // Which output we are writing to? ( 000 - 3 Bits )
  private val outIdxWidth = log2Ceil(outCount + 1)
  // What is the system word size in terms of bytes
  // ( 64 Bits / 8 Bits = 8 Bytes )
  private val xlenBytes = xLen / 8
  // Which byte are we addressing ? ( 000 - 3 Bits )
  private val lgXlenBytes = log2Ceil(xlenBytes)
  // How much large the Cache is int terms of bytes ?
  // ( 128 Bits / 8 Bits = 16 Bytes )
  private val beatBytes = cacheDataBits / 8
  // How many words we get pear burst?
  // ( 16 Bytes / 8 Bytes = 2 )
  private val wordsPerBeat = beatBytes / xlenBytes
  // How many bytes we need to process at a time?
  private val laneWidth = math.max(1, log2Ceil(wordsPerBeat))
  // Which byte are processing from the beat?
  private val beatOffsetBits = log2Ceil(beatBytes)

  require(isPow2(xlenBytes), s"xLen bytes must be a power of two, got ${xlenBytes}")
  require(cacheDataBits % xLen == 0, s"cacheDataBits must be a multiple of xLen (${cacheDataBits} % ${xLen})")
  require(wordsPerBeat > 0, "wordsPerBeat must be > 0")
  require(nRows > 0 && nCols > 0, "mesh dimensions must be > 0")
  require(maxK > 0, "maxK must be > 0")
  require(precision * nRows <= xLen, s"left edge payload exceeds xLen (${precision * nRows} > ${xLen})")
  require(precision * nCols <= xLen, s"top edge payload exceeds xLen (${precision * nCols} > ${xLen})")
  require(accumPrec <= xLen, s"output element width exceeds xLen (${accumPrec} > ${xLen})")

  val core = Module(new SystolicArrayCore(
    precision = precision,
    nRows = nRows,
    nCols = nCols,
    maxK = maxK
  ))

  val flatOut = Wire(Vec(outCount, UInt(accumPrec.W)))
  for (i <- 0 until nRows) {
    for (j <- 0 until nCols) {
      flatOut(i * nCols + j) := core.io.out(i)(j)
    }
  }

  val leftVec = Wire(Vec(nRows, UInt(precision.W)))
  val topVec = Wire(Vec(nCols, UInt(precision.W)))
  val leftBufs = Reg(Vec(2, Vec(maxK, UInt(xLen.W))))
  val topBufs = Reg(Vec(2, Vec(maxK, UInt(xLen.W))))

  val s_idle :: s_prime_buffer :: s_core_start :: s_run :: s_wait_prefetch :: s_req_put :: s_wait_put :: s_resp :: Nil = Enum(8)
  val state = RegInit(s_idle)

  val fill_idle :: fill_req_left :: fill_wait_left :: fill_req_top :: fill_wait_top :: Nil = Enum(5)
  val fillState = RegInit(fill_idle)

  val aBase = RegInit(0.U(xLen.W))
  val bBase = RegInit(0.U(xLen.W))
  val cBase = RegInit(0.U(xLen.W))
  val configured = RegInit(false.B)

  val kRemaining = RegInit(0.U(xLen.W))
  val currChunkK = RegInit(0.U(kWidth.W))
  val currChunkBaseIdx = RegInit(0.U(xLen.W))
  val nextChunkK = RegInit(0.U(kWidth.W))
  val nextChunkBaseIdx = RegInit(0.U(xLen.W))
  val activeBufSel = RegInit(0.U(1.W))
  val fillBufSel = RegInit(0.U(1.W))
  val fillIdx = RegInit(0.U(kWidth.W))
  val fillK = RegInit(0.U(kWidth.W))
  val fillBaseIdx = RegInit(0.U(xLen.W))
  val fillTargetPrefetch = RegInit(false.B)
  val feedIdx = RegInit(0.U(kWidth.W))
  val currBufReady = RegInit(false.B)
  val prefetchedValid = RegInit(false.B)
  val outIdx = RegInit(0.U(outIdxWidth.W))
  val chunkClear = RegInit(true.B)

  val respRd = RegInit(0.U(5.W))
  val respData = RegInit(0.U(xLen.W))

  val feedWordIdx = WireDefault(0.U(kWidth.W))
  when(currChunkK =/= 0.U) {
    feedWordIdx := Mux(feedIdx < currChunkK, feedIdx, currChunkK - 1.U)
  }

  val currentLeftWord = WireDefault(0.U(xLen.W))
  val currentTopWord = WireDefault(0.U(xLen.W))
  currentLeftWord := leftBufs(activeBufSel)(feedWordIdx)
  currentTopWord := topBufs(activeBufSel)(feedWordIdx)

  for (i <- 0 until nRows) {
    leftVec(i) := currentLeftWord(precision * (i + 1) - 1, precision * i)
  }
  for (j <- 0 until nCols) {
    topVec(j) := currentTopWord(precision * (j + 1) - 1, precision * j)
  }

  core.io.cmdValid := false.B
  core.io.k := currChunkK
  core.io.cmdClear := chunkClear
  core.io.inValid := false.B
  core.io.in_left := 0.U.asTypeOf(core.io.in_left)
  core.io.in_top := 0.U.asTypeOf(core.io.in_top)

  val (tlOut, edgesOut) = outer.atlNode.out(0)

  val fillAbsIdx = fillBaseIdx + fillIdx
  val fillLeftAddr = aBase + fillAbsIdx * xlenBytes.U
  val fillTopAddr = bBase + fillAbsIdx * xlenBytes.U
  val writeAddr = cBase + outIdx * xlenBytes.U
  val fillLeftLane = if (wordsPerBeat > 1) fillLeftAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  val fillTopLane = if (wordsPerBeat > 1) fillTopAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)
  val writeLane = if (wordsPerBeat > 1) writeAddr(beatOffsetBits - 1, lgXlenBytes) else 0.U(laneWidth.W)

  val getFillLeftBits = edgesOut.Get(
    fromSource = 0.U,
    toAddress = fillLeftAddr,
    lgSize = lgXlenBytes.U
  )._2
  val getFillTopBits = edgesOut.Get(
    fromSource = 0.U,
    toAddress = fillTopAddr,
    lgSize = lgXlenBytes.U
  )._2
  val putDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    putDataWords(w) := Mux(writeLane === w.U, flatOut(outIdx).pad(xLen), 0.U)
  }
  val putWord = putDataWords.asUInt
  val putBits = edgesOut.Put(
    fromSource = 0.U,
    toAddress = writeAddr,
    lgSize = lgXlenBytes.U,
    data = putWord
  )._2
  val readDataWords = Wire(Vec(wordsPerBeat, UInt(xLen.W)))
  for (w <- 0 until wordsPerBeat) {
    readDataWords(w) := tlOut.d.bits.data((w + 1) * xLen - 1, w * xLen)
  }

  tlOut.a.valid := false.B
  tlOut.a.bits := getFillLeftBits
  tlOut.d.ready := false.B

  io.cmd.ready := (state === s_idle)
  io.resp.valid := (state === s_resp)
  io.resp.bits.rd := respRd
  io.resp.bits.data := respData

  when(io.cmd.fire) {
    respRd := io.cmd.bits.inst.rd
    val funct = io.cmd.bits.inst.funct

    when(funct === 0.U) {
      // funct=0: configure stream bases (rs1=A packed stream base, rs2=B packed stream base)
      aBase := io.cmd.bits.rs1
      bBase := io.cmd.bits.rs2
      configured := true.B
      respData := 0.U
      // Jump to the start and expect funct = 1
      state := s_resp
    }.elsewhen(funct === 1.U) {
      // funct=1: run one output tile (rs1=C output base, rs2=total K; chunked in hardware)
      cBase := io.cmd.bits.rs1
      val initialChunkSize = Mux(io.cmd.bits.rs2 > maxK.U, maxK.U, io.cmd.bits.rs2)
      kRemaining := io.cmd.bits.rs2
      currChunkK := initialChunkSize(kWidth - 1, 0)
      currChunkBaseIdx := 0.U
      nextChunkK := 0.U
      nextChunkBaseIdx := 0.U
      activeBufSel := 0.U
      fillBufSel := 0.U
      fillIdx := 0.U
      fillK := initialChunkSize(kWidth - 1, 0)
      fillBaseIdx := 0.U
      fillTargetPrefetch := false.B
      feedIdx := 0.U
      currBufReady := io.cmd.bits.rs2 === 0.U
      prefetchedValid := false.B
      fillState := Mux(io.cmd.bits.rs2 === 0.U, fill_idle, fill_req_left)
      outIdx := 0.U
      chunkClear := true.B
      respData := Mux(configured, 0.U, 1.U)
      // If we did not do funct = 0 before jump back to idle
      state := Mux(configured, Mux(io.cmd.bits.rs2 === 0.U, s_core_start, s_prime_buffer), s_resp)
    }.elsewhen(funct === 2.U) {
      // funct=2: status query
      respData := Cat(0.U((xLen - 2).W), core.io.busy, core.io.done)
      state := s_resp
    }.otherwise {
      respData := "hDEAD".U
      state := s_resp
    }
  }

  when(state === s_prime_buffer) {
    when(currBufReady) {
      state := s_core_start
    }
  }

  when(state === s_core_start) {
    core.io.cmdValid := true.B
    when(core.io.cmdReady) {
      feedIdx := 0.U
      state := s_run
    }
  }

  when(state === s_run && feedIdx < currChunkK) {
    core.io.inValid := true.B
    core.io.in_left := leftVec
    core.io.in_top := topVec

    when(core.io.inReady) {
      val nextIdx = feedIdx + 1.U
      feedIdx := Mux(nextIdx === currChunkK, currChunkK, nextIdx)
    }
  }

  when(state === s_run && fillState === fill_idle) {
    val remainingAfterCurrent = kRemaining - currChunkK
    val canPrefetch = remainingAfterCurrent =/= 0.U && !prefetchedValid
    when(canPrefetch) {
      val nextChunkSize = Mux(remainingAfterCurrent > maxK.U, maxK.U, remainingAfterCurrent)
      nextChunkK := nextChunkSize(kWidth - 1, 0)
      nextChunkBaseIdx := currChunkBaseIdx + currChunkK
      fillBufSel := activeBufSel ^ 1.U
      fillIdx := 0.U
      fillK := nextChunkSize(kWidth - 1, 0)
      fillBaseIdx := currChunkBaseIdx + currChunkK
      fillTargetPrefetch := true.B
      fillState := fill_req_left
    }
  }

  when(state === s_run) {
    when(core.io.done) {
      val remAfter = kRemaining - currChunkK
      kRemaining := remAfter
      chunkClear := false.B
      when(remAfter === 0.U) {
        outIdx := 0.U
        state := s_req_put
      }.elsewhen(prefetchedValid) {
        activeBufSel := activeBufSel ^ 1.U
        currChunkK := nextChunkK
        currChunkBaseIdx := nextChunkBaseIdx
        prefetchedValid := false.B
        feedIdx := 0.U
        state := s_core_start
      }.otherwise {
        state := s_wait_prefetch
      }
    }
  }

  when(state === s_wait_prefetch) {
    when(prefetchedValid) {
      activeBufSel := activeBufSel ^ 1.U
      currChunkK := nextChunkK
      currChunkBaseIdx := nextChunkBaseIdx
      prefetchedValid := false.B
      feedIdx := 0.U
      state := s_core_start
    }
  }

  when(fillState === fill_req_left) {
    tlOut.a.valid := true.B
    tlOut.a.bits := getFillLeftBits
    when(tlOut.a.fire) {
      fillState := fill_wait_left
    }
  }

  when(fillState === fill_wait_left) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      leftBufs(fillBufSel)(fillIdx) := readDataWords(fillLeftLane)
      val nextIdx = fillIdx + 1.U
      when(nextIdx === fillK) {
        fillIdx := 0.U
        fillState := fill_req_top
      }.otherwise {
        fillIdx := nextIdx
        fillState := fill_req_left
      }
    }
  }

  when(fillState === fill_req_top) {
    tlOut.a.valid := true.B
    tlOut.a.bits := getFillTopBits
    when(tlOut.a.fire) {
      fillState := fill_wait_top
    }
  }

  when(fillState === fill_wait_top) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      topBufs(fillBufSel)(fillIdx) := readDataWords(fillTopLane)
      val nextIdx = fillIdx + 1.U
      when(nextIdx === fillK) {
        fillIdx := 0.U
        fillState := fill_idle
        when(fillTargetPrefetch) {
          prefetchedValid := true.B
        }.otherwise {
          currBufReady := true.B
        }
      }.otherwise {
        fillIdx := nextIdx
        fillState := fill_req_top
      }
    }
  }

  when(state === s_req_put) {
    tlOut.a.valid := true.B
    tlOut.a.bits := putBits
    when(tlOut.a.fire) {
      state := s_wait_put
    }
  }

  when(state === s_wait_put) {
    tlOut.d.ready := true.B
    when(tlOut.d.fire) {
      when(outIdx === (outCount - 1).U) {
        respData := 0.U
        state := s_resp
      }.otherwise {
        outIdx := outIdx + 1.U
        state := s_req_put
      }
    }
  }

  when(io.resp.fire) {
    state := s_idle
  }
}

class SystolicArrayOSRoCC(
  opcodes: OpcodeSet,
  val precision: Int = 16,
  val nRows: Int = 2,
  val nCols: Int = 2,
  val maxK: Int = 64
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new SystolicArrayImpl(
    this
  )
  override val atlNode = TLClientNode(
    Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1("SystolicArrayRoCC"))))
  )
}
