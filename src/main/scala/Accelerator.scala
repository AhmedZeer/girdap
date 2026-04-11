package toyrocc

import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.{Config, Parameters}


class AttnAccel4x4 extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val osm = LazyModule(new OnlineSoftmax(
        intPrecision = 32,
        fracPrecision = 32,
        useBFloat16Output = true,
        opcodes = OpcodeSet.custom0
      )(p))
      osm
    },
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayWSRoCC(
        precision = 16,
        nRows = 4,
        nCols = 4,
        maxK = 256,
        useBFloat16Input = true,
        useBFloat16Output = true,
        fixedPointFracBits = 8,
        accumBits = 64,
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})


class SoftmaxAccel extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val osm = LazyModule(new OnlineSoftmax(
        intPrecision = 32,
        fracPrecision = 32,
        useBFloat16Output = true,
        opcodes = OpcodeSet.custom0
      )(p))
      osm
    })
})

class MatmulAccel2x2OS extends Config((site, here, up) => {
  case BuildRoCC => List(
  (p: Parameters) => {
    val sa = LazyModule(new SystolicArrayOSRoCC(
      precision = 16,
      nRows = 2,
      nCols = 2,
      maxK = 64,
      opcodes = OpcodeSet.custom1,
    )(p))
    sa
  }
  )
})


class MatmulAccel4x4OS extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayOSRoCC(
        precision = 16,
        nRows = 4,
        nCols = 4,
        maxK = 64,
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})

class MatmulAccel2x2WS extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayWSRoCC(
        precision = 16,
        nRows = 2,
        nCols = 2,
        maxK = 256,
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})

class MatmulAccel4x4WS extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayWSRoCC(
        precision = 16,
        nRows = 4,
        nCols = 4,
        maxK = 256,
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})

class MatmulAccel4x4WSBF16 extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayWSRoCC(
        precision = 16,
        nRows = 4,
        nCols = 4,
        maxK = 256,
        useBFloat16Input = true,
        useBFloat16Output = true,
        fixedPointFracBits = 8,
        accumBits = 64,
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})

class MatmulAccel2x2 extends MatmulAccel2x2OS
class MatmulAccel4x4 extends MatmulAccel4x4OS