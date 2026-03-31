package chipyard

import org.chipsalliance.cde.config.Config
import toyrocc.{MatmulAccel, SoftmaxAccel}

class SoftmaxAccelConfig extends Config(
  new SoftmaxAccel ++
    new freechips.rocketchip.rocket.With1TinyCore ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class MatmulAccelConfig extends Config(
  new MatmulAccel ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)
