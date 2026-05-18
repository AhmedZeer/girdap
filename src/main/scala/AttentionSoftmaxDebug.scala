package girdap

import chisel3._
import chisel3.util._
import scala.math.BigDecimal.RoundingMode

import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.Parameters

class AttentionSoftmaxDebugImpl(
  outer: AttentionSoftmaxDebugRoCC
)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters {

  private val nCols = outer.nCols
  private val maxK = outer.maxK
  private val fixedPointFracBits = outer.fixedPointFracBits
  private val accumPrec = outer.accumBits
  private val softmaxIntPrecision = outer.softmaxIntPrecision
  private val softmaxFracPrecision = outer.softmaxFracPrecision
  private val useSoftmaxExpLut = outer.useSoftmaxExpLut
  private val lutEntries = outer.softmaxRecipLutEntries
  private val lutBits = 64
  private val expLutEntries = outer.softmaxExpLutEntries
  private val expLutRange = outer.softmaxExpLutRange
  private val scoreFracBits = 2 * fixedPointFracBits
  private val softBitWidth = softmaxIntPrecision + softmaxFracPrecision
  private val expLutIndexBits = log2Ceil(expLutEntries)
  private val expLutInputShift = scoreFracBits + log2Ceil(expLutRange) - expLutIndexBits
  private val minScoreFixed = (-(BigInt(1) << (accumPrec - 1))).S(accumPrec.W)
  private val kWidth = log2Ceil(maxK + 1)
  private val colCountWidth = log2Ceil(nCols + 1)
  private val tileBaseWidth = log2Ceil(maxK + nCols + 1)
  private val scoreTiles = maxK / nCols
  private val tileIdxWidth = log2Ceil(scoreTiles)

  require(xLen == 64, s"software ABI expects RV64 xLen, got $xLen")
  require(nCols == 8, "attention softmax debug mirrors the 8-wide attention softmax")
  require(accumPrec <= xLen, "score/debug values must fit in xLen")
  require(isPow2(lutEntries), "softmax reciprocal LUT entries must be a power of two")
  require(isPow2(expLutEntries), "softmax exp LUT entries must be a power of two")
  require(isPow2(expLutRange), "softmax exp LUT range must be a power of two")
  require(expLutInputShift >= 0, "softmax exp LUT index needs more entries than score fixed-point resolution supports")
  require(softmaxFracPrecision >= log2Ceil(lutEntries), "softmax frac precision too small")
  require(maxK % nCols == 0, "debug tile count expects full-width tiles")
  require(isPow2(scoreTiles), "debug tile count expects power-of-two tile count")

  private val s_idle :: s_p1_load :: s_p1_update :: s_p2_load :: s_p2_store :: s_resp :: Nil = Enum(6)
  private val state = RegInit(s_idle)

  private val vecLen = RegInit(0.U(kWidth.W))
  private val configured = RegInit(false.B)
  private val tileBase = RegInit(0.U(tileBaseWidth.W))
  private val activeCols = RegInit(0.U(colCountWidth.W))
  private val rowMaxReg = RegInit(minScoreFixed)
  private val rowDenomReg = RegInit(0.U(softBitWidth.W))
  private val invDenomReg = RegInit(0.U(softBitWidth.W))
  private val softLatchedScores = Reg(Vec(nCols, SInt(accumPrec.W)))

  private val scoreStore = RegInit(VecInit(Seq.fill(maxK)(0.S(accumPrec.W))))
  private val diffStore = Reg(Vec(maxK, SInt(accumPrec.W)))
  private val expStore = Reg(Vec(maxK, UInt(softBitWidth.W)))
  private val probStore = Reg(Vec(maxK, UInt(softBitWidth.W)))
  private val tileMaxStore = Reg(Vec(scoreTiles, SInt(accumPrec.W)))
  private val tileDenomStore = Reg(Vec(scoreTiles, UInt(softBitWidth.W)))

  private val respRd = RegInit(0.U(5.W))
  private val respData = RegInit(0.U(xLen.W))

  private val rowMaxRead = rowMaxReg
  private val rowDenomRead = rowDenomReg

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
  private val softSubBase = Mux(state === s_p2_store, rowMaxRead, softGlobalMaxFixed)
  private val softInvSum = reciprocalFixed(rowDenomRead)

  private val softVecFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  private val softProbFixed = Wire(Vec(nCols, UInt(softBitWidth.W)))
  private val softDiffFixed = Wire(Vec(nCols, SInt(accumPrec.W)))
  for (i <- 0 until nCols) {
    val laneValid = i.U < activeCols
    softDiffFixed(i) := softLatchedScores(i) - softSubBase
    val expValue = Wire(UInt(softBitWidth.W))
    if (useSoftmaxExpLut) {
      expValue := expFixedLut(softDiffFixed(i))
    } else {
      val diffBf16 = Module(new SIntFixedToBFloat16(accumPrec, scoreFracBits))
      val exp = Module(new BFloat16Exp)
      val fixed = Module(new BFloat16ToFixed(softmaxIntPrecision, softmaxFracPrecision))
      diffBf16.io.in := softDiffFixed(i)(accumPrec - 1, 0).asSInt
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

  io.cmd.ready := state === s_idle
  io.resp.valid := state === s_resp
  io.resp.bits.rd := respRd
  io.resp.bits.data := respData
  io.busy := state =/= s_idle
  io.interrupt := false.B

  private def loadTileScores(): Unit = {
    val remaining = vecLen.pad(tileBaseWidth) - tileBase
    activeCols := Mux(remaining > nCols.U, nCols.U, remaining)(colCountWidth - 1, 0)
    for (c <- 0 until nCols) {
      val idx = tileBase + c.U
      softLatchedScores(c) := Mux(idx < vecLen.pad(tileBaseWidth), scoreStore(idx(kWidth - 1, 0)), minScoreFixed)
    }
  }

  when(io.cmd.fire) {
    respRd := io.cmd.bits.inst.rd
    val funct = io.cmd.bits.inst.funct
    val idx = io.cmd.bits.rs1(kWidth - 1, 0)
    val idxValid = io.cmd.bits.rs1 < vecLen && configured
    when(funct === 20.U) {
      val newLen = io.cmd.bits.rs1(kWidth - 1, 0)
      val lenValid = io.cmd.bits.rs1 =/= 0.U && io.cmd.bits.rs1 <= maxK.U
      vecLen := newLen
      configured := lenValid
      tileBase := 0.U
      activeCols := 0.U
      rowMaxReg := minScoreFixed
      rowDenomReg := 0.U
      invDenomReg := 0.U
      respData := Mux(lenValid, 0.U, 2.U)
      state := s_resp
    }.elsewhen(funct === 21.U) {
      when(idxValid) {
        scoreStore(idx) := io.cmd.bits.rs2.asSInt
        respData := 0.U
      }.otherwise {
        respData := 2.U
      }
      state := s_resp
    }.elsewhen(funct === 22.U) {
      val runValid = configured && vecLen =/= 0.U && vecLen <= maxK.U
      respData := Mux(runValid, 0.U, 2.U)
      when(runValid) {
        tileBase := 0.U
        rowMaxReg := minScoreFixed
        rowDenomReg := 0.U
        invDenomReg := 0.U
        state := s_p1_load
      }.otherwise {
        state := s_resp
      }
    }.elsewhen(funct === 23.U) {
      respData := Mux(idxValid, scoreStore(idx).asUInt, 2.U)
      state := s_resp
    }.elsewhen(funct === 24.U) {
      respData := Mux(idxValid, diffStore(idx).asUInt, 2.U)
      state := s_resp
    }.elsewhen(funct === 25.U) {
      respData := Mux(idxValid, expStore(idx), 2.U)
      state := s_resp
    }.elsewhen(funct === 26.U) {
      respData := rowDenomReg
      state := s_resp
    }.elsewhen(funct === 27.U) {
      respData := invDenomReg
      state := s_resp
    }.elsewhen(funct === 28.U) {
      respData := Mux(idxValid, probStore(idx), 2.U)
      state := s_resp
    }.elsewhen(funct === 29.U) {
      respData := rowMaxReg.asUInt
      state := s_resp
    }.elsewhen(funct === 30.U) {
      val tileIdx = io.cmd.bits.rs1(tileIdxWidth - 1, 0)
      respData := Mux(io.cmd.bits.rs1 < scoreTiles.U, tileDenomStore(tileIdx), 2.U)
      state := s_resp
    }.elsewhen(funct === 31.U) {
      val tileIdx = io.cmd.bits.rs1(tileIdxWidth - 1, 0)
      respData := Mux(io.cmd.bits.rs1 < scoreTiles.U, tileMaxStore(tileIdx).asUInt, 2.U)
      state := s_resp
    }.otherwise {
      respData := "hDEAD".U
      state := s_resp
    }
  }

  when(state === s_p1_load) {
    loadTileScores()
    state := s_p1_update
  }

  when(state === s_p1_update) {
    val tileIdx = tileBase(tileBaseWidth - 1, log2Ceil(nCols))(tileIdxWidth - 1, 0)
    rowMaxReg := softGlobalMaxFixed
    rowDenomReg := softDenomNext
    tileMaxStore(tileIdx) := softGlobalMaxFixed
    tileDenomStore(tileIdx) := softDenomNext
    val nextTile = tileBase + nCols.U
    when(nextTile >= vecLen.pad(tileBaseWidth)) {
      tileBase := 0.U
      invDenomReg := reciprocalFixed(softDenomNext)
      state := s_p2_load
    }.otherwise {
      tileBase := nextTile
      state := s_p1_load
    }
  }

  when(state === s_p2_load) {
    loadTileScores()
    state := s_p2_store
  }

  when(state === s_p2_store) {
    for (c <- 0 until nCols) {
      val idx = tileBase + c.U
      when(idx < vecLen.pad(tileBaseWidth)) {
        diffStore(idx(kWidth - 1, 0)) := softDiffFixed(c)
        expStore(idx(kWidth - 1, 0)) := softVecFixed(c)
        probStore(idx(kWidth - 1, 0)) := softProbFixed(c)
      }
    }
    val nextTile = tileBase + nCols.U
    when(nextTile >= vecLen.pad(tileBaseWidth)) {
      respData := 0.U
      state := s_resp
    }.otherwise {
      tileBase := nextTile
      state := s_p2_load
    }
  }

  when(io.resp.fire) {
    state := s_idle
  }
}

class AttentionSoftmaxDebugRoCC(
  opcodes: OpcodeSet,
  val nCols: Int = 8,
  val maxK: Int = 256,
  val fixedPointFracBits: Int = 8,
  val accumBits: Int = 64,
  val softmaxIntPrecision: Int = 10,
  val softmaxFracPrecision: Int = 54,
  val softmaxRecipLutEntries: Int = 64,
  val useSoftmaxExpLut: Boolean = true,
  val softmaxExpLutEntries: Int = 2048,
  val softmaxExpLutRange: Int = 16,
  val clientName: String = "AttentionSoftmaxDebugRoCC"
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new AttentionSoftmaxDebugImpl(this)
}
