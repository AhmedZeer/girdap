package atik.memory

import chisel3._
import chisel3.util._

class SramBank(entries: Int, dataBits: Int) extends Module {
  private val addrBits = math.max(1, log2Ceil(entries))
  private val macroDataBits = math.max(32, dataBits)

  val io = IO(new Bundle {
    val wen = Input(Bool())
    val waddr = Input(UInt(addrBits.W))
    val wdata = Input(UInt(dataBits.W))
    val raddr = Input(UInt(addrBits.W))
    val rdata = Output(UInt(dataBits.W))
  })

  private val mem = SyncReadMem(entries, UInt(macroDataBits.W))
  private val addr = Mux(io.wen, io.waddr, io.raddr)
  private val readData = mem.readWrite(addr, io.wdata.pad(macroDataBits), true.B, io.wen)

  io.rdata := readData(dataBits - 1, 0)
}
