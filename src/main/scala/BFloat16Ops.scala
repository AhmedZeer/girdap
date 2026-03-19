package toyrocc

import chisel3._
import chisel3.util._
import hardfloat._

class BFloat16ToFixed(intBits: Int, fracBits: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt((intBits + fracBits).W))
  })
  val sign = io.in(15)
  val exp = io.in(14, 7)
  val mantissa = io.in(6, 0)

  val isZeroOrDenormal = exp === 0.U
  val extMantissa = Mux(isZeroOrDenormal, Cat(0.U(1.W), mantissa), Cat(1.U(1.W), mantissa))

  val totalBits = intBits + fracBits

  val effExp = Mux(isZeroOrDenormal, 1.S(9.W), exp.zext) - 127.S(9.W)
  val shiftAmt = effExp - 7.S + fracBits.S

  val shiftedMag = Wire(UInt(totalBits.W))
  when(shiftAmt >= 0.S) {
    shiftedMag := extMantissa << shiftAmt.asUInt
  }.otherwise {
    shiftedMag := extMantissa >> (-shiftAmt).asUInt
  }

  val outSInt = Wire(SInt(totalBits.W))
  when(sign === 1.U) {
    outSInt := -shiftedMag.zext
  }.otherwise {
    outSInt := shiftedMag.zext
  }

  io.out := outSInt.asUInt
}

class BFloat16Sub extends Module {
  val io = IO(new Bundle {
    val in_1 = Input(UInt(16.W))
    val in_2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val expWidth = 8
  val sigWidth = 9

  val in1Padded = Cat(io.in_1, 0.U(1.W))
  val in2Padded = Cat(io.in_2, 0.U(1.W))

  val recA = recFNFromFN(expWidth, sigWidth, in1Padded)
  val recB = recFNFromFN(expWidth, sigWidth, in2Padded)

  val subFN = Module(new AddRecFN(expWidth, sigWidth))
  subFN.io.subOp := true.B
  subFN.io.a := recA
  subFN.io.b := recB
  subFN.io.roundingMode := "b000".U(3.W)
  subFN.io.detectTininess := 0.U(1.W)

  val outPadded = fNFromRecFN(expWidth, sigWidth, subFN.io.out)
  io.out := outPadded(16, 1)
}

object BFloat16Cmp {
  def max(a: UInt, b: UInt): UInt = {
    val aSign = a(15)
    val bSign = b(15)
    val aMag = a(14, 0)
    val bMag = b(14, 0)

    val isABigger = Mux(
      aSign =/= bSign,
      aSign === 0.U,
      Mux(aSign === 0.U, aMag > bMag, bMag > aMag)
    )
    Mux(isABigger, a, b)
  }
}

class BFloat16Max extends Module {
  val io = IO(new Bundle {
    val in_1 = Input(UInt(16.W))
    val in_2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  io.out := BFloat16Cmp.max(io.in_1, io.in_2)
}

class BFloat16VectorMax(wordsPerBeat: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt((wordsPerBeat * 16).W))
    val out = Output(UInt(16.W))
  })

  val bfloats = VecInit(Seq.tabulate(wordsPerBeat) { i =>
    io.in(16 * (i + 1) - 1, 16 * i)
  })

  io.out := bfloats.reduceTree((a, b) => BFloat16Cmp.max(a, b))
}

class BFloat16Exp extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val sign = io.in(15)
  val exponent = io.in(14, 7)
  val mantissaRaw = io.in(6, 0)

  val isInfNaN = exponent === "hFF".U
  val isZeroMantissa = mantissaRaw === 0.U
  val isNegInf = isInfNaN && isZeroMantissa && (sign === 1.U)

  val mantissa = Mux(exponent === 0.U, 0.U(8.W), Cat(1.U, mantissaRaw))
  val log2eScaled = 369.U(10.W)
  val scaledMantissa = mantissa * log2eScaled
  val paddedMantissa = Cat(scaledMantissa, 0.U(7.W))

  val shift = 133.U - exponent
  val shiftedMantissa = paddedMantissa >> shift
  val signedMantissa = Mux(sign === 0.U(1.W), shiftedMantissa, ~shiftedMantissa)

  val xFrac = signedMantissa(15, 0)
  val xInt = signedMantissa(24, 16)
  val biasedExp = xInt + 127.U
  val finalExp = biasedExp(7, 0)

  val mantissaFrac = xFrac(15, 9)
  val notMantissaFrac = (~mantissaFrac).asUInt

  val alpha = 28.U(7.W)
  val gamma1 = 422.U(9.W)
  val p1 = (mantissaFrac +& gamma1) * alpha * mantissaFrac

  val beta = 56.U(7.W)
  val gamma2 = 278.U(9.W)
  val p2 = (mantissaFrac +& gamma2) * beta * notMantissaFrac

  val isFirstPoly = mantissaFrac(6) === 1.U(1.W)
  val finalMantissa = Mux(isFirstPoly, p1(20, 14), ~(p2(20, 14)))
  val outNormal = Cat(0.U(1.W), finalExp, finalMantissa)
  val outSaturated = "h7F7F".U(16.W)

  io.out := Mux(isNegInf, 0.U, Mux(isInfNaN, outSaturated, outNormal))
}
