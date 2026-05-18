package chipyard

import org.chipsalliance.cde.config.Config
import girdap.{
  FusedOnlineAttention8x8BF16FpgaSafePackerExpLut,
  Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut,
  Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512,
  MatmulAccel8x8BF16FpgaSafe,
  SoftmaxAccel
}


class FusedOnlineAttention8x8BF16FpgaSafePackerExpLutConfig extends Config(
  new FusedOnlineAttention8x8BF16FpgaSafePackerExpLut ++
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

class Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512Config extends Config(
  new Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512 ++
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

class MatmulAccel8x8BF16FpgaSafeConfig extends Config(
  new MatmulAccel8x8BF16FpgaSafe ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)
