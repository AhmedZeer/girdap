package toyrocc

import chisel3._
import chisel3.util._
import hardfloat._

import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tilelink.{
  TLNode, TLIdentityNode, TLClientNode, TLMasterParameters, TLMasterPortParameters
}

class BFloat16ToFixed(intBits: Int, fracBits: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt((intBits + fracBits).W))
  })
  val sign = io.in(15)
  val exp = io.in(14, 7)
  val mantissa = io.in(6, 0)

  // UQ1.7 representation
  val is_zero_or_denormal = exp === 0.U
  val ext_mantissa = Mux(is_zero_or_denormal, Cat(0.U(1.W), mantissa), Cat(1.U(1.W), mantissa))

  val totalBits = intBits + fracBits

  // 1. Calculate signed exponent to avoid unsigned wrap-around
  // Denormals (exp == 0) have an effective exponent of -126 in IEEE 754
  val eff_exp = Mux(is_zero_or_denormal, 1.S(9.W), exp.zext) - 127.S(9.W)

  // 2. Align to the Fixed-Point fractional width
  // BFloat16 value = ext_mantissa * 2^(-7) * 2^(eff_exp)
  // Target Fixed-point value = Int * 2^(-fracBits)
  // Shift required = eff_exp - 7 + fracBits
  val shift_amt = eff_exp - 7.S + fracBits.S

  val shifted_mag = Wire(UInt(totalBits.W))

  // 3. Shift left or right depending on the exponent and fracBits
  when(shift_amt >= 0.S) {
    shifted_mag := ext_mantissa << shift_amt.asUInt
  } .otherwise {
    shifted_mag := ext_mantissa >> (-shift_amt).asUInt
  }

  // 4. Apply Two's Complement if the sign bit is 1
  val out_sint = Wire(SInt(totalBits.W))
  when(sign === 1.U) {
    out_sint := -shifted_mag.zext
  } .otherwise {
    out_sint := shifted_mag.zext
  }

  io.out := out_sint.asUInt
}

class BFloat16Sub extends Module {
  val io = IO(new Bundle {
    val in_1 = Input(UInt(16.W))
    val in_2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val expWidth = 8
  // Workaround: Bump precision from 8 to 9 to avoid HardFloat's bit-slice underflow bug
  val sigWidth = 9

  // Pad the 7-bit explicit fraction to 8 bits by appending a 0 at the LSB
  // This turns our 16-bit inputs into 17-bit standard floats
  val in_1_padded = Cat(io.in_1, 0.U(1.W))
  val in_2_padded = Cat(io.in_2, 0.U(1.W))

  val rec_a = recFNFromFN(expWidth, sigWidth, in_1_padded)
  val rec_b = recFNFromFN(expWidth, sigWidth, in_2_padded)

  val subFN = Module(new AddRecFN(expWidth, sigWidth))

  subFN.io.subOp := true.B
  subFN.io.a := rec_a
  subFN.io.b := rec_b
  subFN.io.roundingMode := "b000".U(3.W) // round_near_even
  subFN.io.detectTininess := 0.U(1.W)

  // Output is a 17-bit standard float (1 sign, 8 exp, 8 frac)
  val out_padded = fNFromRecFN(expWidth, sigWidth, subFN.io.out)

  // Truncate the padded fractional LSB to return to standard 16-bit BFloat16
  io.out := out_padded(16, 1)
}

class BFloat16Max extends Module {
  val io = IO(new Bundle {
    val in_1 = Input(UInt(16.W))
    val in_2 = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  def max(a: UInt, b: UInt): UInt = {
    val a_sign = a(15)
    val b_sign = b(15)
    val a_mag = a(14, 0)
    val b_mag = b(14, 0)

    val is_a_bigger = Mux(a_sign =/= b_sign,
      a_sign === 0.U,
      Mux(a_sign === 0.U, a_mag > b_mag, b_mag > a_mag)
    )

    Mux(is_a_bigger, a, b)
  }

  io.out := max(io.in_1, io.in_2)
}

class BFloat16VectorMax(wordsPerBeat: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt((wordsPerBeat * 16).W))
    val out = Output(UInt(16.W))
  })

  def max(a: UInt, b: UInt): UInt = {
    val a_sign = a(15)
    val b_sign = b(15)
    val a_mag = a(14, 0)
    val b_mag = b(14, 0)

    val is_a_bigger = Mux(a_sign =/= b_sign,
      a_sign === 0.U,
      Mux(a_sign === 0.U, a_mag > b_mag, b_mag > a_mag)
    )

    Mux(is_a_bigger, a, b)
  }

  val bfloats = VecInit(Seq.tabulate(wordsPerBeat) { i =>
    io.in(16 * (i + 1) - 1, 16 * i)
  })

  // neat chisel
  io.out := bfloats.reduceTree((a, b) => max(a, b))
}

class BFloat16Exp extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val sign = io.in(15)
  val exponent = io.in(14, 7)
  val mantissa_raw = io.in(6, 0)

  val is_inf_nan = exponent === "hFF".U
  val is_zero_mantissa = mantissa_raw === 0.U
  val is_neg_inf = is_inf_nan && is_zero_mantissa && (sign === 1.U)

  // If subnormal, fill with zeroes.
  // else append 1 to the mantissa
  val mantissa = Mux(exponent === 0.U, 0.U(8.W), Cat(1.U, mantissa_raw))

  // Convert log2e to a fixed-point
  // representation : UQ2.8
  // Unsigined fixed-point with 2 bits for integer and 8 for fraction.
  val log2e_scaled = 369.U(10.W)

  // Mantissa : UQ1.7
  // log2e    : UQ2.8
  // Mantissa * log2e : UQ3.15
  val scaled_mantissa = mantissa * log2e_scaled

  // Padded Mantissa : UQ9.16
  // We are adding more precision.
  val padded_mantissa = Cat(scaled_mantissa, 0.U(7.W))

  // Considering the biggest number representable
  // by BFloat16 we get : 3.39 x 10^38
  // So maximum exponentiation result is :
  // e^x = 3.39 x 10^38
  // Solving for x, we get x ~ 88.72
  // BFloat(x) = sign(0), exp(133), mantissa(??)
  // RESULT : the exp of our input can be 133 at maximum.
  val shift = 133.U - exponent

  // Shifted Mantissa : UQ9.16
  val shifted_mantissa = padded_mantissa >> shift
  val signed_mantissa = Mux(sign === 0.U(1.W), shifted_mantissa, ~shifted_mantissa)

  // 16 bits for fraction
  val x_frac = signed_mantissa(15, 0)
  // 9 bits for integer
  val x_int = signed_mantissa(24, 16)
  val biased_exp = x_int + 127.U

  // UQ8.0
  val final_exp = biased_exp(7, 0)

  // UQ0.7
  val mantissa_frac = x_frac(15, 9)
  val not_mantissa_frac = (~mantissa_frac).asUInt

  // alpha   * 2^7 = 0.21875  * 128  = 28
  // beta    * 2^7 = 0.4375   * 128  = 56
  // gamma_1 * 2^7 = 3.296875 * 128  = 422
  // gamma_2 * 2^7 = 2.171875 * 128  = 278

  // When mantissa belongs to [0, 0.5)
  val alpha = 28.U(7.W)      // UQ0.7
  val gamma_1 = 422.U(9.W)   // UQ2.7

  // UQ3.21
  // `+&` carry add operation to increase the
  // integer bitwidth.
  val p_1 = (mantissa_frac +& gamma_1) * alpha * mantissa_frac

  // When mantissa belongs to [0.5, 1)
  val beta = 56.U(7.W)      // UQ0.7
  val gamma_2 = 278.U(9.W)  // UQ2.7

  // UQ3.21
  val p_2 = (mantissa_frac +& gamma_2) * beta * not_mantissa_frac

  val isFirstPoly = mantissa_frac(6) === 1.U(1.W)

  // Extract the fraction parts from p_1 & p_2 (20, 0)
  val final_mantissa = Mux(isFirstPoly, p_1(20, 14), ~(p_2(20, 14)))
  val out_normal = Cat(0.U(1.W), final_exp, final_mantissa)
  val out_saturated = "h7F7F".U(16.W) // max finite BF16

  io.out := Mux(is_neg_inf, 0.U, Mux(is_inf_nan, out_saturated, out_normal))
}


class BFloat16ExpAccumulator(opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new BFloat16ExpAccumulatorModuleImp(this)
  override val atlNode = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1("BFloat16ExpAccumulatorRoCC")))))
}

class BFloat16ExpAccumulatorModuleImp(outer: BFloat16ExpAccumulator)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters
  with HasL1CacheParameters {

  val cacheParams = tileParams.dcache.get

  // 64-bit cache bus / 16-bit BFloat16  = 4 elements per beat
  // 256-bit cache bus / 16-bit BFloat16 = 16 elements per beat
  val wordsPerBeat = cacheDataBits / 16

  val addr = Reg(UInt(coreMaxAddrBits.W))
  val elements_left = Reg(UInt(xLen.W))
  val accumulator_fixedp = Reg(UInt(xLen.W))
  val resp_rd = Reg(chiselTypeOf(io.resp.bits.rd))

  val s_idle :: s_acq :: s_gnt :: s_resp :: Nil = Enum(4)
  val state = RegInit(s_idle)

  val (tl_out, edgesOut) = outer.atlNode.out(0)

  io.cmd.ready := (state === s_idle)
  io.resp.valid := (state === s_resp)
  io.resp.bits.rd := resp_rd
  io.resp.bits.data := accumulator_fixedp

  tl_out.a.valid := (state === s_acq)

  // The size of data we are expecting
  val bytesPerBeat = cacheDataBits / 8
  val bytesPerBeatU = bytesPerBeat.U

  tl_out.a.bits := edgesOut.Get(
    fromSource = 0.U,
    toAddress = addr,
    lgSize = log2Ceil(bytesPerBeat).U)._2
  tl_out.d.ready := (state === s_gnt)

  when (io.cmd.fire) {
    addr := io.cmd.bits.rs1
    elements_left := io.cmd.bits.rs2
    resp_rd := io.cmd.bits.inst.rd
    accumulator_fixedp := 0.U
    state := Mux(io.cmd.bits.rs2 === 0.U, s_resp, s_acq)
    printf(p"[ACCUM EXP HW] CMD FIRE! Received rd: ${resp_rd} || addr: ${addr} ... \n")
  }

  when (tl_out.a.fire) { state := s_gnt }


  // Tuple4 from edgesOut.count
  val (first, last, done, beat_count) = edgesOut.count(tl_out.d)

  when (tl_out.d.fire) {
    // Slice beat into chunks
    val bfloat_inputs = VecInit(Seq.tabulate(wordsPerBeat) { i =>
      tl_out.d.bits.data(16 * (i + 1) - 1, 16 * i)
    })

    val exp_outputs = Wire(Vec(wordsPerBeat, UInt(16.W)))
    val expModules = Seq.fill(wordsPerBeat)(Module(new BFloat16Exp))
    for(i <- 0 until wordsPerBeat){
      expModules(i).io.in := bfloat_inputs(i)
      exp_outputs(i) := expModules(i).io.out
    }

    // Convert BFloat16 outputs to 64-bit Integers for hardware accumulation
    val fixed_vals = exp_outputs.map { bf16 =>
      val exp = bf16(14, 7)
      val mantissa = bf16(6, 0)

      val implied_bit = Mux(exp === 0.U, 0.U(1.W), 1.U(1.W))
      // UQ1.7
      val fraction = Cat(implied_bit, mantissa).pad(64)

      // Ones digit must be shifted by (32 - 7 = 25) times.
      // Exponent is biased, exp - 127 + 25 = exp - 102
      val is_shift_left = exp > 102.U

      // We will make sure that we shift
      // maximum of 64 bits.
      val left_shift_amt = exp - 102.U
      val right_shift_amt = 102.U - exp

      Mux(exp === 0.U, 0.U(64.W),
        Mux(is_shift_left, fraction << left_shift_amt(5, 0), fraction >> right_shift_amt(5, 0))
      )
    }

    // Mask out invalid elements
    val masked_vals = fixed_vals.zipWithIndex.map { case (int_val, i) =>
      Mux(i.U < elements_left, int_val, 0.U(64.W))
    }

    val beat_sum = masked_vals.reduce(_ + _)
    accumulator_fixedp := accumulator_fixedp + beat_sum
    addr := addr + bytesPerBeatU

    val elements_processed = Mux(elements_left > wordsPerBeat.U, wordsPerBeat.U, elements_left)
    val next_elements = elements_left - elements_processed
    elements_left := next_elements
    printf(p"[ACCUM EXP HW] Got the data!")

    when (done) {
      state := Mux(next_elements === 0.U, s_resp, s_acq)
    }
  }

  when (io.resp.fire) {
    state := s_idle
    printf(p"[ACCUM EXP HW] Back to idle.")
  }
}

class OnlineSoftmaxImpl(intPrecision:Int = 12, fracPrecision: Int = 20, outer: OnlineSoftmax)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters
  with HasL1CacheParameters {

  // required for tilelink
  val cacheParams = tileParams.dcache.get
  // n_elements we get per request from tilelink
  val wordsPerBeat = cacheDataBits / 16
  // -inf
  val minBf16 = "hFF80".U(16.W)
  val bitWidth = intPrecision + fracPrecision

  // States
  // s_idle -> Fetch the next instruction.
  // s_acq  -> Ask for the next element.
  // s_findmax -> Find the current new maximum from latched elements & prev max
  // s_exp -> Exponentiate the thing
  // s_resp -> Update the accumulation register, increase the total processed elements,
  //           if we consumed all the data finish.
  val s_idle :: s_acq :: s_gnt :: s_findmax :: s_exp :: s_reduce :: s_resp :: Nil = Enum(7)
  val state = RegInit(s_idle)

  // Operational Registers
  // UQintPrecision.fracPrecision, i.e UQ12.20
  val accum = RegInit(0.U(bitWidth.W))

  // Corresponds for m_i and m_i-1 in Online Softmax paper.
  val maxCurr = RegInit(minBf16) // currMax = max(maxVec(input), prevMax)
                                 // Updated when finding the maximum
  val maxPrev = RegInit(minBf16) // Updated when reducing


  // Vec Operation Modules
  val vecSubs = Seq.fill(wordsPerBeat)(Module(new BFloat16Sub))
  val vecExps = Seq.fill(wordsPerBeat)(Module(new BFloat16Exp))
  val vecFixedPs = Seq.fill(wordsPerBeat)(Module(
    new BFloat16ToFixed(intBits = intPrecision, fracBits = fracPrecision)
  ))
  val vecFixedWires = Wire(Vec(wordsPerBeat, UInt((intPrecision + fracPrecision).W)))

  // Reduction Modules
  val maxSub = Module(new BFloat16Sub)
  val maxExp = Module(new BFloat16Exp)
  val maxFixedP = Module(
    new BFloat16ToFixed(intBits = intPrecision, fracBits = fracPrecision)
  )

  // Input Command Registers
  val arraySize = RegInit(0.U(xLen.W)) // rs2
  val addr = RegInit(0.U(xLen.W))      // rs1
  val resp_rd = RegInit(0.U(5.W))      // rd

  val bytesPerBeat = (cacheDataBits / 8).U // typically 8-bytes
  val procElements = RegInit(0.U(xLen.W)) // starts from 0 ends at arraySize
  val latchedData  = RegInit(0.U(cacheDataBits.W)) // updated at cmd.fire

  val bfloat_inputs = VecInit(Seq.tabulate(wordsPerBeat) { i =>
    val raw_data = latchedData(16 * (i + 1) - 1, 16 * i)

    // Handle out-of-bound
    val is_garbage = procElements + i.U >= arraySize
    Mux(is_garbage, minBf16, raw_data)
  })

  // Max Modules
  val globalMax = Module(new BFloat16Max)
  val vecMax = Module(new BFloat16VectorMax(wordsPerBeat))
  vecMax.io.in := bfloat_inputs.asUInt
  globalMax.io.in_1 := vecMax.io.out
  globalMax.io.in_2 := maxPrev

  maxSub.io.in_1 := maxPrev
  maxSub.io.in_2 := globalMax.io.out
  maxExp.io.in := maxSub.io.out
  maxFixedP.io.in := maxExp.io.out

  // safe clamping because the exp approximation
  // doesnt return exactly 1.0 (1.02 i.e) Slight errors and
  // deviations will eventually lead to big errors.
  val one = (1.U(bitWidth.W) << fracPrecision)(bitWidth - 1, 0) // UQ1.0 in bitWidth
  val maxFixedPClamped = Mux(maxFixedP.io.out > one, one, maxFixedP.io.out)

  // UQ32.32 * UQ32.32
  val prodAccumFull = accum * maxFixedPClamped
  val prodAccum = (prodAccumFull >> fracPrecision)(bitWidth - 1, 0)

  for(i <- 0 until wordsPerBeat){
    vecSubs(i).io.in_1 := bfloat_inputs(i)
    vecSubs(i).io.in_2 := globalMax.io.out
    vecExps(i).io.in := vecSubs(i).io.out
    vecFixedPs(i).io.in := vecExps(i).io.out
    vecFixedWires(i) := vecFixedPs(i).io.out
  }
  val vecSum = vecFixedWires.reduceTree(_ + _)

  // TileLink Stuff
  val (tl_out, edgesOut) = outer.atlNode.out(0)

  io.cmd.ready := (state === s_idle)
  io.resp.valid := (state === s_resp)
  io.resp.bits.data := accum
  io.resp.bits.rd := resp_rd

  tl_out.a.valid := (state === s_acq)
  tl_out.a.bits := edgesOut.Get(
    fromSource = 0.U,
    toAddress = addr,
    lgSize = log2Ceil(cacheDataBits / 8).U
  )._2
  tl_out.d.ready := (state === s_gnt)

  when(io.cmd.fire){
    arraySize := io.cmd.bits.rs2
    addr := io.cmd.bits.rs1
    resp_rd := io.cmd.bits.inst.rd

    accum := 0.U
    procElements := 0.U
    maxPrev := minBf16

    state := s_acq
  }

  when(tl_out.a.fire){
    addr := addr + bytesPerBeat
    state := s_gnt
  }

  when(tl_out.d.fire){
    latchedData := tl_out.d.bits.data
    state := s_findmax
  }

  when(state === s_findmax){
    maxCurr := globalMax.io.out
    state := s_exp
  }

  when(state === s_exp){
    state := s_reduce
  }

  when(state === s_reduce){
    maxPrev := maxCurr
    accum := vecSum + prodAccum
    val nextProcElements = procElements + wordsPerBeat.U
    procElements := nextProcElements

    when(nextProcElements >= arraySize){
      state := s_resp
    }.otherwise{
      state := s_acq
    }
  }

  when(io.resp.fire){
    state := s_idle
  }
}

class OnlineSoftmax(intPrecision:Int = 12, fracPrecision:Int = 20, opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new OnlineSoftmaxImpl(intPrecision = intPrecision, fracPrecision = fracPrecision, this)
  override val atlNode = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLMasterParameters.v1("OnlineSoftmaxRoCC")))))
}

class WithToyRoCC extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val expAccum = LazyModule(new OnlineSoftmax(
        intPrecision = 32,
        fracPrecision = 32,
        opcodes = OpcodeSet.custom0
      )(p))
      expAccum
    })
})
