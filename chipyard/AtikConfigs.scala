package chipyard

import atik.AtikParams
import atik.rocc.WithAtikRoCC
import org.chipsalliance.cde.config.Config

class Atik2x2RoCCConfig extends Config(
  new WithAtikRoCC(AtikParams(meshRows = 2, meshCols = 2)) ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class Atik2x2MatmulOnlyRoCCConfig extends Config(
  new WithAtikRoCC(AtikParams(meshRows = 2, meshCols = 2, enableAttention = false)) ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class Atik4x4RoCCConfig extends Config(
  new WithAtikRoCC(AtikParams(meshRows = 4, meshCols = 4)) ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class Atik8x8RoCCConfig extends Config(
  new WithAtikRoCC(AtikParams(meshRows = 8, meshCols = 8)) ++
    new freechips.rocketchip.subsystem.WithoutTLMonitors ++
    new freechips.rocketchip.rocket.WithNHugeCores(1) ++
    new chipyard.config.WithSystemBusWidth(128) ++
    new chipyard.config.AbstractConfig
)

class AtikRoCCConfig extends Atik8x8RoCCConfig
