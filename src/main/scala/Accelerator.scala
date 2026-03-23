package toyrocc

import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import org.chipsalliance.cde.config.{Config, Parameters}

class WithToyRoCC extends Config((site, here, up) => {
  case BuildRoCC => List(
    /*
    (p: Parameters) => {
      val osm = LazyModule(new OnlineSoftmax(
        intPrecision = 32,
        fracPrecision = 32,
        opcodes = OpcodeSet.custom0
      )(p))
      osm
    },
     */
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayRoCC(
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
