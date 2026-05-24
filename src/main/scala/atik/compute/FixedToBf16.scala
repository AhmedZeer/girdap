package atik.compute

import atik._
import chisel3._
import chisel3.util._

class FixedToBf16(params: AtikParams, inBits: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(SInt(inBits.W))
    val out = Output(UInt(params.elemBits.W))
  })

  private val expBias = 127
  private val maxFiniteExp = 254
  private val mantissaBits = 7
  private val maxShift = inBits + mantissaBits
  private val shiftWidth = math.max(1, log2Ceil(maxShift + 1))

  private val sign = io.in(inBits - 1)
  private val inUInt = io.in.asUInt
  private val magnitude = Mux(sign, (~inUInt).asUInt + 1.U, inUInt)
  private val isZero = magnitude === 0.U

  private val leadingZeros = PriorityEncoder(Reverse(magnitude))
  private val msbIdx = (inBits - 1).U - leadingZeros
  private val unbiasedExp = msbIdx.zext - params.fixedFracBits.S((msbIdx.getWidth + 2).W)
  private val biasedExp = unbiasedExp + expBias.S((msbIdx.getWidth + 3).W)

  private val rightShiftAmt = Wire(UInt(shiftWidth.W))
  private val leftShiftAmt = Wire(UInt(shiftWidth.W))
  rightShiftAmt := 0.U
  leftShiftAmt := 0.U

  private val shiftLeft = msbIdx < mantissaBits.U
  when(shiftLeft) {
    leftShiftAmt := mantissaBits.U - msbIdx
  }.otherwise {
    rightShiftAmt := msbIdx - mantissaBits.U
  }

  private val normalizedSig = Wire(UInt((mantissaBits + 1).W))
  when(shiftLeft) {
    normalizedSig := (magnitude << leftShiftAmt)(mantissaBits, 0)
  }.otherwise {
    normalizedSig := (magnitude >> rightShiftAmt)(mantissaBits, 0)
  }

  private val normal = Cat(sign, biasedExp(7, 0).asUInt, normalizedSig(mantissaBits - 1, 0))
  private val maxFinite = Cat(sign, maxFiniteExp.U(8.W), Fill(mantissaBits, 1.U(1.W)))

  io.out := Mux(
    isZero || biasedExp <= 0.S,
    0.U,
    Mux(biasedExp > maxFiniteExp.S, maxFinite, normal)
  )
}
