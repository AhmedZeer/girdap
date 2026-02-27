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

class VEXPCmd extends Bundle {
  val rd = UInt(5.W)
}

class VEXPResp(XLen: Int) extends Bundle {
  val data = UInt(XLen.W)
  val rd = UInt(5.W)
}

class VEXPCore(XLen: Int) extends Module {
  val io = IO(new Bundle {
    val cmd = Flipped(DecoupledIO(new VEXPCmd()))
    val resp = DecoupledIO(new VEXPResp(XLen))
    val busy = Output(Bool())
  })
  val res = RegInit(0.U(XLen.W))
  val res_rd = RegInit(0.U(5.W))

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
      printf(p"[VEXP HW] CMD FIRE! Received rd: ${io.cmd.bits.rd}\n")
    }
  }

  when(state === s_proc){
    res := 30.U(XLen.W)
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
