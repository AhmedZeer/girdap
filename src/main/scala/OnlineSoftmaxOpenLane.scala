package girdap

import chisel3._
import chisel3.util._
import scala.math.BigDecimal.RoundingMode

class OnlineSoftmaxFastStandalone extends Module {
  private val intPrecision = 12
  private val fracPrecision = 20
  private val lutEntries = 64
  private val lutBits = 64
  private val xLen = 64
  private val cacheDataBits = 64
  private val bitWidth = intPrecision + fracPrecision
  private val wordsPerBeat = cacheDataBits / 16
  private val outElemBits = 16
  private val beatBytes = cacheDataBits / 8
  private val minBf16 = "hFF80".U(16.W)
  private val lutIndexBits = log2Ceil(lutEntries)

  require(isPow2(lutEntries) && lutEntries >= 2)
  require(fracPrecision >= lutIndexBits - 1)

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

  private val sIdle :: sReqRead :: sWaitRead :: sFindMax :: sExp :: sReduce :: sReqWrite :: sWaitWrite :: sResp :: Nil = Enum(9)
  private val state = RegInit(sIdle)

  private val accum = RegInit(0.U(bitWidth.W))
  private val maxCurr = RegInit(minBf16)
  private val maxPrev = RegInit(minBf16)
  private val arraySize = RegInit(0.U(xLen.W))
  private val inBase = RegInit(0.U(xLen.W))
  private val inAddr = RegInit(0.U(xLen.W))
  private val outAddr = RegInit(0.U(xLen.W))
  private val procElements = RegInit(0.U(xLen.W))
  private val latchedData = RegInit(0.U(cacheDataBits.W))
  private val invSum = RegInit(0.U(bitWidth.W))
  private val doWrite = RegInit(false.B)
  private val respRd = RegInit(0.U(5.W))
  private val respData = RegInit(0.U(xLen.W))

  private val vecSubs = Seq.fill(wordsPerBeat)(Module(new BFloat16Sub))
  private val vecExps = Seq.fill(wordsPerBeat)(Module(new BFloat16Exp))
  private val vecFixed = Seq.fill(wordsPerBeat)(Module(new BFloat16ToFixed(intPrecision, fracPrecision)))
  private val vecNormOut = Seq.fill(wordsPerBeat)(Module(new UIntFixedToBFloat16(bitWidth, fracPrecision)))
  private val maxSub = Module(new BFloat16Sub)
  private val maxExp = Module(new BFloat16Exp)
  private val maxFixed = Module(new BFloat16ToFixed(intPrecision, fracPrecision))
  private val globalMax = Module(new BFloat16Max)
  private val vecMax = Module(new BFloat16VectorMax(wordsPerBeat))

  private val lut = VecInit(Seq.tabulate(lutEntries) { i =>
    val mant = BigDecimal(1) + (BigDecimal(i) + BigDecimal(0.5)) / BigDecimal(lutEntries)
    val recip = BigDecimal(1) / mant
    val scaled = (recip * BigDecimal(2).pow(lutBits - 1)).setScale(0, RoundingMode.HALF_UP).toBigInt
    scaled.U(lutBits.W)
  })

  private val bfloatInputs = VecInit(Seq.tabulate(wordsPerBeat) { i =>
    val raw = latchedData(16 * (i + 1) - 1, 16 * i)
    Mux(procElements + i.U >= arraySize, minBf16, raw)
  })

  vecMax.io.in := bfloatInputs.asUInt
  globalMax.io.in_1 := vecMax.io.out
  globalMax.io.in_2 := maxPrev

  maxSub.io.in_1 := maxPrev
  maxSub.io.in_2 := globalMax.io.out
  maxExp.io.in := maxSub.io.out
  maxFixed.io.in := maxExp.io.out

  private val one = (1.U(bitWidth.W) << fracPrecision)(bitWidth - 1, 0)
  private val maxFixedClamped = Mux(maxFixed.io.out > one, one, maxFixed.io.out)
  private val prodAccumFull = accum * maxFixedClamped
  private val prodAccum = (prodAccumFull >> fracPrecision)(bitWidth - 1, 0)

  private val subBase = Mux(doWrite, maxPrev, globalMax.io.out)
  private val vecFixedWires = Wire(Vec(wordsPerBeat, UInt(bitWidth.W)))
  private val vecOutWires = Wire(Vec(wordsPerBeat, UInt(outElemBits.W)))
  for (i <- 0 until wordsPerBeat) {
    vecSubs(i).io.in_1 := bfloatInputs(i)
    vecSubs(i).io.in_2 := subBase
    vecExps(i).io.in := vecSubs(i).io.out
    vecFixed(i).io.in := vecExps(i).io.out
    vecFixedWires(i) := vecFixed(i).io.out

    val normFull = vecFixedWires(i) * invSum
    val normFixed = (normFull >> fracPrecision)(bitWidth - 1, 0)
    vecNormOut(i).io.in := normFixed
    vecOutWires(i) := Mux(procElements + i.U < arraySize, vecNormOut(i).io.out, 0.U)
  }
  private val vecSum = vecFixedWires.reduceTree(_ + _)

  private val sumNonZero = accum.orR
  private val msb = (bitWidth - 1).U - PriorityEncoder(Reverse(accum))
  private val shift = Mux(sumNonZero, msb - fracPrecision.U, 0.U)
  private val mant = accum >> shift
  private val lutIndex = mant(fracPrecision, fracPrecision - lutIndexBits + 1)
  private val lutVal = lut(lutIndex)
  private val lutScaled = lutVal >> (lutBits - 1 - fracPrecision)
  private val invRaw = lutScaled >> shift
  private val invSumCalc = Mux(sumNonZero, invRaw.pad(bitWidth)(bitWidth - 1, 0), 0.U)

  private val validBytes = Wire(Vec(beatBytes, Bool()))
  for (byte <- 0 until beatBytes) {
    validBytes(byte) := procElements + (byte / 2).U < arraySize
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
    when(io.cmdFunct === 0.U) {
      doWrite := false.B
      arraySize := io.cmdRs2
      inBase := io.cmdRs1
      inAddr := io.cmdRs1
      procElements := 0.U
      accum := 0.U
      maxPrev := minBf16
      respData := 0.U
      state := Mux(io.cmdRs2 === 0.U, sResp, sReqRead)
    }.otherwise {
      doWrite := true.B
      outAddr := io.cmdRs1
      inAddr := inBase
      procElements := 0.U
      invSum := invSumCalc
      respData := 0.U
      state := Mux(arraySize === 0.U, sResp, sReqRead)
    }
  }

  when(state === sReqRead) {
    io.memReqValid := true.B
    io.memReqAddr := inAddr
    io.memReqMask := Fill(beatBytes, 1.U(1.W))
    when(io.memReqReady) {
      state := sWaitRead
    }
  }

  when(state === sWaitRead) {
    io.memRespReady := true.B
    when(io.memRespValid) {
      latchedData := io.memRespData
      state := Mux(doWrite, sExp, sFindMax)
    }
  }

  when(state === sFindMax) {
    maxCurr := globalMax.io.out
    state := sExp
  }

  when(state === sExp) {
    state := Mux(doWrite, sReqWrite, sReduce)
  }

  when(state === sReduce) {
    maxPrev := maxCurr
    accum := vecSum + prodAccum
    inAddr := inAddr + beatBytes.U
    val nextProcElements = procElements + wordsPerBeat.U
    procElements := nextProcElements
    when(nextProcElements >= arraySize) {
      respData := (vecSum + prodAccum).pad(xLen)
      state := sResp
    }.otherwise {
      state := sReqRead
    }
  }

  when(state === sReqWrite) {
    io.memReqValid := true.B
    io.memReqWrite := true.B
    io.memReqAddr := outAddr
    io.memReqWdata := vecOutWires.asUInt
    io.memReqMask := validBytes.asUInt
    when(io.memReqReady) {
      state := sWaitWrite
    }
  }

  when(state === sWaitWrite) {
    io.memRespReady := true.B
    when(io.memRespValid) {
      outAddr := outAddr + beatBytes.U
      inAddr := inAddr + beatBytes.U
      val nextProcElements = procElements + wordsPerBeat.U
      procElements := nextProcElements
      when(nextProcElements >= arraySize) {
        state := sResp
      }.otherwise {
        state := sReqRead
      }
    }
  }

  when(io.respValid && io.respReady) {
    state := sIdle
  }
}

class OnlineSoftmaxFastOpenLaneTop extends Module {
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
    val memReqWdata = Output(UInt(64.W))
    val memReqMask = Output(UInt(8.W))
    val memRespValid = Input(Bool())
    val memRespReady = Output(Bool())
    val memRespData = Input(UInt(64.W))
    val memRespWriteAck = Input(Bool())
    val busy = Output(Bool())
  })

  val dut = Module(new OnlineSoftmaxFastStandalone)
  dut.io <> io
}

object EmitOnlineSoftmaxFastOpenLane extends App {
  val targetDir =
    if (args.nonEmpty) args(0)
    else "/home/ubuntu/firesim_build_disk/openlane_softmax_fast/src"

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new OnlineSoftmaxFastOpenLaneTop,
    Array("--target-dir", targetDir),
    Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    )
  )
}
