package atik

import chisel3.util.log2Ceil

case class AtikParams(
  meshRows: Int = 8,
  meshCols: Int = 8,
  elemBits: Int = 16,
  fixedBits: Int = 24,
  fixedFracBits: Int = 11,
  accumBits: Int = 64,
  softmaxBits: Int = 32,
  softmaxFracBits: Int = 24,
  expLutEntries: Int = 256,
  expLutRange: Int = 16,
  recipLutEntries: Int = 256,
  addrBits: Int = 64,
  xLen: Int = 64,
  memDataBits: Int = 128,
  dmaBeatsBits: Int = 16,
  counterCount: Int = 13,
  matmulKt: Int = 16,
  enableAttention: Boolean = true
) {
  require(meshRows > 0 && meshCols > 0, "Atik mesh dimensions must be positive")
  require(Set(2, 4, 8).contains(meshRows), "Atik supports 2x2, 4x4, and 8x8 meshes")
  require(meshRows == meshCols, "Atik currently assumes square meshes")
  require(matmulKt >= meshCols, "matmul KT should be at least the mesh column count")
  require(counterCount >= 13, "Atik counter bank must include the manifest counter set")
  require(elemBits == 16, "Atik external tensor elements are BF16")
  require(fixedBits > fixedFracBits, "fixed-point format needs integer bits")
  require(accumBits >= fixedBits * 2, "accumulator should hold fixed-point products")
  require(accumBits > fixedFracBits * 2, "accumulator needs integer room above product fractional bits")
  require(memDataBits % elemBits == 0, "memory beat must contain an integer number of BF16 elements")
  require(memDataBits % xLen == 0, "memory beat must contain an integer number of xLen words")
  require(memDataBits == 64 || memDataBits == 128, "descriptor parser currently supports 64b or 128b beats")
  require(isPow2(expLutEntries), "exp LUT entries must be a power of two")
  require(isPow2(recipLutEntries), "reciprocal LUT entries must be a power of two")

  val accumFracBits: Int = fixedFracBits * 2
  val bytesPerElem: Int = elemBits / 8
  val beatBytes: Int = memDataBits / 8
  val wordsPerBeat: Int = memDataBits / xLen
  val elemsPerBeat: Int = memDataBits / elemBits
  val descriptorBytes: Int = 80
  val descriptorBeats: Int = (descriptorBytes + beatBytes - 1) / beatBytes

  def counterIdxBits: Int = math.max(1, log2Ceil(counterCount))
  def dmaBeatIdxBits: Int = math.max(1, dmaBeatsBits)

  private def isPow2(x: Int): Boolean = x > 0 && ((x & (x - 1)) == 0)
}
