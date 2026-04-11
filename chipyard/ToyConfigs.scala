package chipyard

import org.chipsalliance.cde.config.Config
import toyrocc.{AttnAccel4x4, MatmulAccel2x2, MatmulAccel2x2WS, MatmulAccel4x4, MatmulAccel4x4WS, MatmulAccel4x4WSBF16, SoftmaxAccel}

class AttnAccel4x4WS256Config extends Config(
  new AttnAccel4x4 ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(256) ++
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