package girdap

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

class FusedAttnAccel4x4 extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val attn = LazyModule(new FusedAttentionWSRoCC(
        precision = 16,
        nRows = 4,
        nCols = 4,
        maxK = 128,
        fixedPointFracBits = 8,
        accumBits = 32,
        softmaxIntPrecision = 12,
        softmaxFracPrecision = 20,
        numTLSourceIds = 2,
        opcodes = OpcodeSet.custom0
      )(p))
      attn
    }
  )
})

class FusedOnlineAttention8x8BF16FpgaSafe extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val attn = LazyModule(new FpgaSafeOnlineAttention8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 512,
        fixedPointFracBits = 8,
        accumBits = 64,
        softmaxIntPrecision = 32,
        softmaxFracPrecision = 32,
        numTLSourceIds = 2,
        clientName = "FusedOnlineAttention8x8BF16FpgaSafeRoCC",
        opcodes = OpcodeSet.custom1
      )(p))
      attn
    }
  )
})

class FusedOnlineAttention8x8BF16FpgaSafePacker extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val attn = LazyModule(new FpgaSafeOnlineAttention8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 256,
        fixedPointFracBits = 8,
        accumBits = 64,
        softmaxIntPrecision = 10,
        softmaxFracPrecision = 54,
        softmaxRecipLutEntries = 64, // 1024 
        numTLSourceIds = 2,
        enablePacker = true,
        clientName = "FusedOnlineAttention8x8BF16FpgaSafePackerRoCC",
        opcodes = OpcodeSet.custom1
      )(p))
      attn
    }
  )
})

class FusedOnlineAttention8x8BF16FpgaSafePackerExpLut extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val attn = LazyModule(new FpgaSafeOnlineAttention8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 512,
        fixedPointFracBits = 8,
        accumBits = 64,
        softmaxIntPrecision = 10,
        softmaxFracPrecision = 54,
        softmaxRecipLutEntries = 64,
        useSoftmaxExpLut = true,
        softmaxExpLutEntries = 2048,
        softmaxExpLutRange = 16,
        numTLSourceIds = 2,
        enablePacker = true,
        clientName = "FusedOnlineAttention8x8BF16FpgaSafePackerExpLutRoCC",
        opcodes = OpcodeSet.custom1
      )(p))
      attn
    }
  )
})

class FusedOnlineAttention8x8BF16FpgaSafePackerExpLut512 extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val attn = LazyModule(new FpgaSafeOnlineAttention8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 512,
        fixedPointFracBits = 8,
        accumBits = 64,
        softmaxIntPrecision = 10,
        softmaxFracPrecision = 54,
        softmaxRecipLutEntries = 64,
        useSoftmaxExpLut = true,
        softmaxExpLutEntries = 2048,
        softmaxExpLutRange = 16,
        numTLSourceIds = 2,
        enablePacker = true,
        clientName = "FusedOnlineAttention8x8BF16FpgaSafePackerExpLutRoCC",
        opcodes = OpcodeSet.custom1
      )(p))
      attn
    }
  )
})

class FusedOnlineAttention8x8BF16FpgaSafePackerExpLutDebug extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val attn = LazyModule(new FpgaSafeOnlineAttention8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 256,
        fixedPointFracBits = 8,
        accumBits = 64,
        softmaxIntPrecision = 10,
        softmaxFracPrecision = 54,
        softmaxRecipLutEntries = 64,
        useSoftmaxExpLut = true,
        softmaxExpLutEntries = 2048,
        softmaxExpLutRange = 16,
        numTLSourceIds = 2,
        enablePacker = true,
        enableIntermediateDebug = true,
        clientName = "FusedOnlineAttention8x8BF16FpgaSafePackerExpLutDebugRoCC",
        opcodes = OpcodeSet.custom1
      )(p))
      attn
    }
  )
})

class Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayFpgaSafe8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 256,
        fixedPointFracBits = 8,
        accumBits = 64,
        numTLSourceIds = 2,
        clientName = "Matmul8x8BF16FpgaSafeRoCC",
        opcodes = OpcodeSet.custom0,
      )(p))
      sa
    },
    (p: Parameters) => {
      val attn = LazyModule(new FpgaSafeOnlineAttention8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 256,
        fixedPointFracBits = 8,
        accumBits = 64,
        softmaxIntPrecision = 10,
        softmaxFracPrecision = 54,
        softmaxRecipLutEntries = 64,
        useSoftmaxExpLut = true,
        softmaxExpLutEntries = 2048,
        softmaxExpLutRange = 16,
        numTLSourceIds = 2,
        enablePacker = true,
        clientName = "FusedOnlineAttention8x8BF16FpgaSafePackerExpLutRoCC",
        opcodes = OpcodeSet.custom1
      )(p))
      attn
    }
  )
})

class Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512 extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayFpgaSafe8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 512,
        fixedPointFracBits = 8,
        accumBits = 64,
        numTLSourceIds = 2,
        clientName = "Matmul8x8BF16FpgaSafeRoCC",
        opcodes = OpcodeSet.custom0,
      )(p))
      sa
    },
    (p: Parameters) => {
      val attn = LazyModule(new FpgaSafeOnlineAttention8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 512,
        fixedPointFracBits = 8,
        accumBits = 64,
        softmaxIntPrecision = 10,
        softmaxFracPrecision = 54,
        softmaxRecipLutEntries = 64,
        useSoftmaxExpLut = true,
        softmaxExpLutEntries = 2048,
        softmaxExpLutRange = 16,
        numTLSourceIds = 2,
        enablePacker = true,
        clientName = "FusedOnlineAttention8x8BF16FpgaSafePackerExpLutRoCC",
        opcodes = OpcodeSet.custom1
      )(p))
      attn
    }
  )
})

class AttentionSoftmaxDebug extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val dbg = LazyModule(new AttentionSoftmaxDebugRoCC(
        nCols = 8,
        maxK = 256,
        fixedPointFracBits = 8,
        accumBits = 64,
        softmaxIntPrecision = 10,
        softmaxFracPrecision = 54,
        softmaxRecipLutEntries = 64,
        useSoftmaxExpLut = true,
        softmaxExpLutEntries = 2048,
        softmaxExpLutRange = 16,
        clientName = "AttentionSoftmaxDebugRoCC",
        opcodes = OpcodeSet.custom1
      )(p))
      dbg
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

class MatmulAccel4x4WSBF16NoAPrefetch extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayWSNoAPrefetchRoCC(
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

class MatmulAccel4x4BF16FpgaSafe extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayFpgaSafeRoCC(
        precision = 16,
        nRows = 4,
        nCols = 4,
        maxK = 256,
        fixedPointFracBits = 8,
        accumBits = 64,
        numTLSourceIds = 2,
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})

class MatmulAccel8x8BF16FpgaSafe extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayFpgaSafe8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 256,
        fixedPointFracBits = 8,
        accumBits = 64,
        numTLSourceIds = 2,
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})

class FusedMatmulSoftmax8x8BF16FpgaSafe extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayFpgaSafe8x8RoCC(
        precision = 16,
        nRows = 8,
        nCols = 8,
        maxK = 256,
        fixedPointFracBits = 8,
        accumBits = 64, // Keeps it simple for the ABI side.
        numTLSourceIds = 2,
        applyRowSoftmax = true,
        softmaxIntPrecision = 32,
        softmaxFracPrecision = 32,
        clientName = "FusedMatmulSoftmax8x8FpgaSafeRoCC",
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})

class MatmulAccel2x2WSBF16 extends Config((site, here, up) => {
  case BuildRoCC => List(
    (p: Parameters) => {
      val sa = LazyModule(new SystolicArrayWSRoCC(
        precision = 16,
        nRows = 2,
        nCols = 2,
        maxK = 256,
        useBFloat16Input = true,
        useBFloat16Output = true,
        fixedPointFracBits = 8,
        accumBits = 32,
        opcodes = OpcodeSet.custom1,
      )(p))
      sa
    }
  )
})

class MatmulAccel2x2 extends MatmulAccel2x2OS
class MatmulAccel4x4 extends MatmulAccel4x4OS
