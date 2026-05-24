package atik.util

import chisel3._
import chisel3.util.Cat
import chisel3.util.Fill

object FixedPointUtil {
  def resizeSInt(value: SInt, width: Int): SInt = {
    val inWidth = value.getWidth
    if (inWidth == width) {
      value
    } else if (inWidth > width) {
      value(width - 1, 0).asSInt
    } else {
      Cat(Fill(width - inWidth, value(inWidth - 1)), value.asUInt).asSInt
    }
  }

  def resizeUInt(value: UInt, width: Int): UInt = {
    val inWidth = value.getWidth
    if (inWidth == width) {
      value
    } else if (inWidth > width) {
      value(width - 1, 0)
    } else {
      value.pad(width)
    }
  }

  def narrowSInt(value: SInt, width: Int): SInt = resizeSInt(value, width)

  def narrowUInt(value: UInt, width: Int): UInt = resizeUInt(value, width)

  def signedProductToAccum(a: SInt, b: SInt, fracBits: Int, accumBits: Int): SInt = {
    val product = a * b
    resizeSInt((product >> fracBits).asSInt, accumBits)
  }
}
