package girdap

import chisel3._
import chisel3.util._
import scala.math.BigDecimal.RoundingMode

import org.chipsalliance.cde.config.{Parameters, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tilelink.{TLClientNode, TLMasterParameters, TLMasterPortParameters}

class OnlineSoftmaxImpl(
  intPrecision: Int = 12,
  fracPrecision: Int = 20,
  lutEntries: Int = 256,
  lutBits: Int = 64,
  outer: OnlineSoftmax
)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters
  with HasL1CacheParameters {

  val cacheParams = tileParams.dcache.get
  val useBFloat16Output = outer.useBFloat16Output
  val bitWidth = intPrecision + fracPrecision
  val wordsPerBeat = cacheDataBits / 16
  val outElemBits = if (useBFloat16Output) 16 else bitWidth
  val outElemBytes = outElemBits / 8
  val outWordsPerBeat = cacheDataBits / outElemBits
  val outBeatsPerInBeat = wordsPerBeat / outWordsPerBeat
  val writeBeatIdxWidth = math.max(1, log2Ceil(outBeatsPerInBeat))
  val minBf16 = "hFF80".U(16.W)
  val lutIndexBits = log2Ceil(lutEntries)

  require(isPow2(lutEntries) && lutEntries >= 2, "lutEntries must be a power of 2 and >= 2")
  require(cacheDataBits % 16 == 0, "cacheDataBits must be a multiple of the BF16 lane width")
  require(cacheDataBits % outElemBits == 0, "cacheDataBits must be a multiple of the output element width")
  require(wordsPerBeat % outWordsPerBeat == 0, "output beat ratio must be integral")
  require(fracPrecision >= lutIndexBits - 1, "fracPrecision too small for LUT index bits")

  val s_idle :: s_acq :: s_gnt :: s_findmax :: s_exp :: s_reduce :: s_put :: s_put_ack :: s_resp :: Nil = Enum(9)
  val state = RegInit(s_idle)

  val accum = RegInit(0.U(bitWidth.W))
  val maxCurr = RegInit(minBf16)
  val maxPrev = RegInit(minBf16)

  val vecSubs = Seq.fill(wordsPerBeat)(Module(new BFloat16Sub))
  val vecExps = Seq.fill(wordsPerBeat)(Module(new BFloat16Exp))
  val vecFixedPs = Seq.fill(wordsPerBeat)(Module(
    new BFloat16ToFixed(intBits = intPrecision, fracBits = fracPrecision)
  ))
  val vecFixedWires = Wire(Vec(wordsPerBeat, UInt((intPrecision + fracPrecision).W)))
  val vecNormFixedWires = Wire(Vec(wordsPerBeat, UInt(bitWidth.W)))
  val vecOutWires = Wire(Vec(wordsPerBeat, UInt(outElemBits.W)))

  val maxSub = Module(new BFloat16Sub)
  val maxExp = Module(new BFloat16Exp)
  val maxFixedP = Module(
    new BFloat16ToFixed(intBits = intPrecision, fracBits = fracPrecision)
  )

  val lut = VecInit(Seq.tabulate(lutEntries) { i =>
    val mant = BigDecimal(1) + (BigDecimal(i) + BigDecimal(0.5)) / BigDecimal(lutEntries)
    val recip = BigDecimal(1) / mant
    val scaled = (recip * BigDecimal(2).pow(lutBits - 1)).setScale(0, RoundingMode.HALF_UP).toBigInt
    scaled.U(lutBits.W)
  })

  val arraySize = RegInit(0.U(xLen.W))
  val inBase = RegInit(0.U(xLen.W))
  val inAddr = RegInit(0.U(xLen.W))
  val outAddr = RegInit(0.U(xLen.W))
  val respRd = RegInit(0.U(5.W))
  val invSum = RegInit(0.U(bitWidth.W))
  val doWrite = RegInit(false.B)
  val writeBeatIdx = RegInit(0.U(writeBeatIdxWidth.W))
  val outBuf = Reg(Vec(wordsPerBeat, UInt(outElemBits.W)))

  val bytesPerBeat = (cacheDataBits / 8).U
  val outBeatBytes = bytesPerBeat
  val procElements = RegInit(0.U(xLen.W))
  val latchedData = RegInit(0.U(cacheDataBits.W))

  val bfloatInputs = VecInit(Seq.tabulate(wordsPerBeat) { i =>
    val rawData = latchedData(16 * (i + 1) - 1, 16 * i)
    val isGarbage = procElements + i.U >= arraySize
    Mux(isGarbage, minBf16, rawData)
  })

  val globalMax = Module(new BFloat16Max)
  val vecMax = Module(new BFloat16VectorMax(wordsPerBeat))
  vecMax.io.in := bfloatInputs.asUInt
  globalMax.io.in_1 := vecMax.io.out
  globalMax.io.in_2 := maxPrev

  maxSub.io.in_1 := maxPrev
  maxSub.io.in_2 := globalMax.io.out
  maxExp.io.in := maxSub.io.out
  maxFixedP.io.in := maxExp.io.out

  val one = (1.U(bitWidth.W) << fracPrecision)(bitWidth - 1, 0)
  val maxFixedPClamped = Mux(maxFixedP.io.out > one, one, maxFixedP.io.out)
  val prodAccumFull = accum * maxFixedPClamped
  val prodAccum = (prodAccumFull >> fracPrecision)(bitWidth - 1, 0)

  val subBase = Mux(doWrite, maxPrev, globalMax.io.out)
  for (i <- 0 until wordsPerBeat) {
    vecSubs(i).io.in_1 := bfloatInputs(i)
    vecSubs(i).io.in_2 := subBase
    vecExps(i).io.in := vecSubs(i).io.out
    vecFixedPs(i).io.in := vecExps(i).io.out
    vecFixedWires(i) := vecFixedPs(i).io.out

    val normFull = vecFixedWires(i) * invSum
    vecNormFixedWires(i) := (normFull >> fracPrecision)(bitWidth - 1, 0)

    if (useBFloat16Output) {
      val conv = Module(new UIntFixedToBFloat16(bitWidth, fracPrecision))
      conv.io.in := vecNormFixedWires(i)
      vecOutWires(i) := conv.io.out
    } else {
      vecOutWires(i) := vecNormFixedWires(i)
    }
  }
  val vecSum = vecFixedWires.reduceTree(_ + _)

  val sumNonZero = accum.orR
  val msb = (bitWidth - 1).U - PriorityEncoder(Reverse(accum))
  val shift = Mux(sumNonZero, msb - fracPrecision.U, 0.U)
  val mant = accum >> shift
  val lutIndex = mant(fracPrecision, fracPrecision - lutIndexBits + 1)
  val lutVal = lut(lutIndex)
  val scaleDown = lutBits - 1 - fracPrecision
  val lutScaled = if (scaleDown >= 0) (lutVal >> scaleDown) else (lutVal << (-scaleDown))
  val invRaw = lutScaled >> shift
  val invSumCalc = Mux(sumNonZero, invRaw.pad(bitWidth)(bitWidth - 1, 0), 0.U)

  val remainingElems = Mux(arraySize >= procElements, arraySize - procElements, 0.U)
  val validPerBeat = Wire(Vec(outBeatsPerInBeat, UInt(log2Ceil(outWordsPerBeat + 1).W)))

  // mask for out-of-bound
  for (b <- 0 until outBeatsPerInBeat) {
    val base = (b * outWordsPerBeat).U
    val rem = Mux(remainingElems > base, remainingElems - base, 0.U)
    validPerBeat(b) := Mux(rem > outWordsPerBeat.U, outWordsPerBeat.U, rem)
  }

  val outBeatVec = VecInit(Seq.tabulate(outBeatsPerInBeat) { b =>
    VecInit(Seq.tabulate(outWordsPerBeat) { j =>
      outBuf(b * outWordsPerBeat + j)
    }).asUInt
  })
  val outMaskVec = VecInit(Seq.tabulate(outBeatsPerInBeat) { b =>
    val valid = validPerBeat(b)
    VecInit(Seq.tabulate(outWordsPerBeat) { j =>
      Fill(outElemBytes, j.U < valid)
    }).asUInt
  })

  val outBeat = outBeatVec(writeBeatIdx)
  val outMask = outMaskVec(writeBeatIdx)

  val (tlOut, edgesOut) = outer.atlNode.out(0)

  io.cmd.ready := (state === s_idle)
  io.resp.valid := (state === s_resp)
  io.resp.bits.data := accum
  io.resp.bits.rd := respRd

  val doRead = (state === s_acq)
  val doPut = (state === s_put)
  val getBits = edgesOut.Get(
    fromSource = 0.U,
    toAddress = inAddr,
    lgSize = log2Ceil(cacheDataBits / 8).U
  )._2
  val putBits = edgesOut.Put(
    fromSource = 0.U,
    toAddress = outAddr,
    lgSize = log2Ceil(cacheDataBits / 8).U,
    data = outBeat,
    mask = outMask
  )._2

  tlOut.a.valid := doRead || doPut
  tlOut.a.bits := Mux(doPut, putBits, getBits)
  tlOut.d.ready := (state === s_gnt) || (state === s_put_ack)

  when(io.cmd.fire) {
    respRd := io.cmd.bits.inst.rd
    val funct = io.cmd.bits.inst.funct

    when(funct === 0.U) {
      doWrite := false.B
      arraySize := io.cmd.bits.rs2
      inBase := io.cmd.bits.rs1
      inAddr := io.cmd.bits.rs1
      procElements := 0.U
      accum := 0.U
      maxPrev := minBf16
      state := s_acq
    }.otherwise {
      doWrite := true.B
      outAddr := io.cmd.bits.rs1
      inAddr := inBase
      procElements := 0.U
      writeBeatIdx := 0.U
      invSum := invSumCalc
      state := s_acq
    }
  }

  when(state === s_acq && tlOut.a.fire) {
    state := s_gnt
  }

  when(state === s_put && tlOut.a.fire) {
    state := s_put_ack
  }

  when(state === s_gnt && tlOut.d.fire) {
    latchedData := tlOut.d.bits.data
    state := Mux(doWrite, s_exp, s_findmax)
  }

  when(state === s_findmax && !doWrite) {
    maxCurr := globalMax.io.out
    state := s_exp
  }

  when(state === s_exp) {
    when(doWrite) {
      for (i <- 0 until wordsPerBeat) {
        val isValid = procElements + i.U < arraySize
        outBuf(i) := Mux(isValid, vecOutWires(i), 0.U(outElemBits.W))
      }
      writeBeatIdx := 0.U
      state := s_put
    }.otherwise {
      state := s_reduce
    }
  }

  when(state === s_reduce && !doWrite) {
    maxPrev := maxCurr
    accum := vecSum + prodAccum
    inAddr := inAddr + bytesPerBeat
    val nextProcElements = procElements + wordsPerBeat.U
    procElements := nextProcElements

    when(nextProcElements >= arraySize) {
      state := s_resp
    }.otherwise {
      state := s_acq
    }
  }

  when(state === s_put_ack && tlOut.d.fire) {
    outAddr := outAddr + outBeatBytes

    val lastBeat = writeBeatIdx === (outBeatsPerInBeat - 1).U
    when(lastBeat) {
      inAddr := inAddr + bytesPerBeat
      val nextProcElements = procElements + wordsPerBeat.U
      procElements := nextProcElements
      writeBeatIdx := 0.U

      when(nextProcElements >= arraySize) {
        state := s_resp
      }.otherwise {
        state := s_acq
      }
    }.otherwise {
      writeBeatIdx := writeBeatIdx + 1.U
      state := s_put
    }
  }

  when(io.resp.fire) {
    state := s_idle
  }
}

class OnlineSoftmax(
  intPrecision: Int = 12,
  fracPrecision: Int = 20,
  lutEntries: Int = 256,
  lutBits: Int = 64,
  val useBFloat16Output: Boolean = false,
  opcodes: OpcodeSet
)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new OnlineSoftmaxImpl(
    intPrecision = intPrecision,
    fracPrecision = fracPrecision,
    lutEntries = lutEntries,
    lutBits = lutBits,
    this
  )
  override val atlNode = TLClientNode(
    Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1("OnlineSoftmaxRoCC"))))
  )
}
