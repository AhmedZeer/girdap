package toyrocc

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.{Parameters, Field, Config}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import freechips.rocketchip.rocket._

class AccumulatorExample(opcodes: OpcodeSet, val n: Int = 4)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new AccumulatorExampleModuleImp(this)
}

class AccumulatorExampleModuleImp(outer: AccumulatorExample)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters {
  val regfile = Mem(outer.n, UInt(xLen.W))
  val busy = RegInit(VecInit(Seq.fill(outer.n){false.B}))

  val cmd = Queue(io.cmd)
  val funct = cmd.bits.inst.funct

  // This is bit slicing. [7:0] for example to get the first 8 bits.
  val addr = cmd.bits.rs2(log2Up(outer.n)-1,0)
  val doWrite = funct === 0.U
  val doRead = funct === 1.U
  val doLoad = funct === 2.U
  val doAccum = funct === 3.U

  // HellaCache responds to multiple requests in an out-of-order manner.
  // Which register address the memory responded to.
  val memRespTag = io.mem.resp.bits.tag(log2Up(outer.n)-1,0)

  // datapath
  // The command bundle contains the value of core's rs1, not the address/index.
  // cmd.rs1 = core.regfile[instr.rs1]
  val addend = cmd.bits.rs1
  val accum = regfile(addr)
  val wdata = Mux(doWrite, addend, accum + addend)

  // .fire indicates that if IO is ready & valid.
  when (cmd.fire && (doWrite || doAccum)) {
    printf("VEXP: Command Received! RD register: %d\n", cmd.bits.inst.rd)
    regfile(addr) := wdata
  }

  // Got the data from L1, proceed.
  when (io.mem.resp.valid) {
    regfile(memRespTag) := io.mem.resp.bits.data
    busy(memRespTag) := false.B
  }

  // Asking for data from L1, wait.
  when (io.mem.req.fire) {
    busy(addr) := true.B
  }

  val doResp = cmd.bits.inst.xd
  val stallReg = busy(addr) // Register is not ready, wait.
  val stallLoad = doLoad && !io.mem.req.ready // L1 is fetching, wait.
  val stallResp = doResp && !io.resp.ready // CPU is not ready, wait.

  // Signal a ready to accept the next command
  cmd.ready := !stallReg && !stallLoad && !stallResp

  // PROC RESPONSE INTERFACE
  io.resp.valid := cmd.valid && doResp && !stallReg && !stallLoad
  // valid response if valid command, need a response, and no stalls
  io.resp.bits.rd := cmd.bits.inst.rd
  // Must respond with the appropriate tag or undefined behavior
  io.resp.bits.data := accum
  // Semantics is to always send out prior accumulator register value

  io.busy := cmd.valid || busy.reduce(_||_)
  // Be busy when have pending memory requests or committed possibility of pending requests
  io.interrupt := false.B
  // Set this true to trigger an interrupt on the processor (please refer to supervisor documentation)

  // MEMORY REQUEST INTERFACE
  io.mem.req.valid := cmd.valid && doLoad && !stallReg && !stallResp
  io.mem.req.bits.addr := addend
  io.mem.req.bits.tag := addr
  io.mem.req.bits.cmd := M_XRD // perform a load (M_XWR for stores)
  io.mem.req.bits.size := log2Ceil(8).U
  io.mem.req.bits.signed := false.B
  io.mem.req.bits.data := 0.U // we're not performing any stores...
  io.mem.req.bits.phys := false.B
  io.mem.req.bits.dprv := cmd.bits.status.dprv
  io.mem.req.bits.dv := cmd.bits.status.dv
  io.mem.req.bits.no_resp := false.B
}

class BFloat16Exp extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val sign = io.in(15)
  val exponent = io.in(14, 7)

  // If subnormal, fill with zeroes.
  // else append 1 to the mantissa
  val mantissa = Mux(exponent === 0.U, 0.U(8.W), Cat(1.U, io.in(6, 0)))

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
  io.out := Cat(0.U(1.W), final_exp, final_mantissa)
}

class VEXPCmd(XLen:Int) extends Bundle {
  val rd = UInt(5.W)
  val rs1 = UInt(XLen.W)
}

class VEXPResp(XLen: Int) extends Bundle {
  val data = UInt(XLen.W)
  val rd = UInt(5.W)
}

class VEXPCore(XLen: Int) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(DecoupledIO(new VEXPCmd(XLen)))
    val resp = DecoupledIO(new VEXPResp(XLen))
    val busy = Output(Bool())
  })
  val res = RegInit(0.U(XLen.W))
  val res_rd = RegInit(0.U(5.W))
  val rs1 = RegInit(0.U(XLen.W))

  // Initial Signals
  val s_idle :: s_proc :: s_finish :: Nil = Enum(3)
  val state = RegInit(s_idle)
  io.cmd.ready := false.B
  io.resp.valid := false.B
  io.resp.bits.rd := 0.U
  io.resp.bits.data := 0.U

  when(state === s_idle){
    io.cmd.ready := true.B
    when(io.cmd.fire){
      state := s_proc
      res_rd := io.cmd.bits.rd
      rs1 := io.cmd.bits.rs1

      // Won't work if non-blocking.
      printf(p"[VEXP HW] CMD FIRE! Received rd: ${res_rd} || rs1: ${rs1}\n")
    }
  }

  when(state === s_proc){


    val bfloat16Inputs = rs1.asTypeOf(Vec(4, UInt(16.W)))
    val out = Wire(Vec(4, UInt(16.W)))
    val vexpBfloat16 = Seq.fill(4)(Module(new BFloat16Exp))

    for(i <- 0 until 4){
      vexpBfloat16(i).io.in := bfloat16Inputs(i)
      out(i) := vexpBfloat16(i).io.out
      printf(p"[VEXP HW] BFloat16 Input  : ${Hexadecimal(bfloat16Inputs(i))}.\n")
      printf(p"[VEXP HW] BFloat16 Output : ${Hexadecimal(out(i))}.\n\n")
    }

    res := out.asUInt
    state := s_finish
    printf(p"[VEXP HW] Processing complete. Moving to finish.\n")
  }

  when(state === s_finish){
    io.resp.bits.data := res
    io.resp.bits.rd := res_rd
    io.resp.valid := true.B
    printf(p"[VEXP HW] Waiting for CPU to accept response... io.resp.ready is ${io.resp.ready}\n")
    when(io.resp.fire){
      state := s_idle
      printf(p"[VEXP HW] RESP FIRE! Sent data 30 back to CPU.\n")
    }
  }

  io.busy := (state =/= s_idle) || io.cmd.valid
}

class VEXP(opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC(opcodes) {
  override lazy val module = new VEXPImpl(this)
}

class VEXPImpl(outer: VEXP)(implicit p: Parameters) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters {
  val core = Module(new VEXPCore(xLen))
  val cmd = Queue(io.cmd)

  core.io.cmd.bits.rd := cmd.bits.inst.rd
  core.io.cmd.bits.rs1 := cmd.bits.rs1
  core.io.cmd.valid := cmd.valid
  cmd.ready := core.io.cmd.ready

  io.resp.bits.data := core.io.resp.bits.data
  io.resp.bits.rd := core.io.resp.bits.rd
  io.resp.valid := core.io.resp.valid
  core.io.resp.ready := io.resp.ready

  io.busy := core.io.busy
  io.interrupt := false.B
  io.mem.req.valid := false.B
}

class WithToyRoCC extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val accumulator = LazyModule(new AccumulatorExample(OpcodeSet.custom0, n = 4)(p))
      accumulator
    },
    (p: Parameters) => {
      val vexp = LazyModule(new VEXP(OpcodeSet.custom1)(p))
      vexp
    })
})
