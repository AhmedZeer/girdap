package chipyard

import org.chipsalliance.cde.config.Config
import toyrocc.{AttentionSoftmaxDebug, AttnAccel4x4, FusedAttnAccel4x4, FusedMatmulSoftmax8x8BF16FpgaSafe, FusedOnlineAttention8x8BF16FpgaSafe, FusedOnlineAttention8x8BF16FpgaSafePacker, FusedOnlineAttention8x8BF16FpgaSafePackerExpLut, FusedOnlineAttention8x8BF16FpgaSafePackerExpLutDebug, Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut, MatmulAccel2x2, MatmulAccel2x2WS, MatmulAccel2x2WSBF16, MatmulAccel4x4, MatmulAccel4x4BF16FpgaSafe, MatmulAccel4x4WS, MatmulAccel4x4WSBF16, MatmulAccel4x4WSBF16NoAPrefetch, MatmulAccel8x8BF16FpgaSafe, SoftmaxAccel}

class AttnAccel4x4WS256Config extends Config(
  new AttnAccel4x4 ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(256) ++
    new chipyard.config.AbstractConfig
)

class FusedAttnAccel4x4Config extends Config(
  new FusedAttnAccel4x4 ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class FusedOnlineAttention8x8BF16FpgaSafeConfig extends Config(
  new FusedOnlineAttention8x8BF16FpgaSafe ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class FusedOnlineAttention8x8BF16FpgaSafePackerConfig extends Config(
  new FusedOnlineAttention8x8BF16FpgaSafePacker ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class FusedOnlineAttention8x8BF16FpgaSafePackerExpLutConfig extends Config(
  new FusedOnlineAttention8x8BF16FpgaSafePackerExpLut ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class FusedOnlineAttention8x8BF16FpgaSafePackerExpLutDebugConfig extends Config(
  new FusedOnlineAttention8x8BF16FpgaSafePackerExpLutDebug ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLutConfig extends Config(
  new Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class AttentionSoftmaxDebugConfig extends Config(
  new AttentionSoftmaxDebug ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class SoftmaxAccel128Config extends Config(
  new SoftmaxAccel ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class SoftmaxAccel256Config extends Config(
  new SoftmaxAccel ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(256) ++
    new chipyard.config.AbstractConfig
)


class MatmulAccel2x2Config extends Config(
  new MatmulAccel2x2 ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)

class MatmulAccel4x4Config extends Config(
  new MatmulAccel4x4 ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel2x2WSConfig extends Config(
  new MatmulAccel2x2WS ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel4x4WSConfig extends Config(
  new MatmulAccel4x4WS ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel4x4WS256Config extends Config(
  new MatmulAccel4x4WS ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(256) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel4x4WSBF16Config extends Config(
  new MatmulAccel4x4WSBF16 ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel4x4WSBF16256Config extends Config(
  new MatmulAccel4x4WSBF16 ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(256) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel4x4WSBF16NoAPrefetchConfig extends Config(
  new MatmulAccel4x4WSBF16NoAPrefetch ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel4x4BF16FpgaSafeConfig extends Config(
  new MatmulAccel4x4BF16FpgaSafe ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel8x8BF16FpgaSafeConfig extends Config(
  new MatmulAccel8x8BF16FpgaSafe ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class FusedMatmulSoftmax8x8BF16FpgaSafeConfig extends Config(
  new FusedMatmulSoftmax8x8BF16FpgaSafe ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccel2x2WSBF16Config extends Config(
  new MatmulAccel2x2WSBF16 ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)
