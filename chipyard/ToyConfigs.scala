package chipyard

import org.chipsalliance.cde.config.Config
import toyrocc.WithToyRoCC

class DummyToyRoCCConfig extends Config(
  new WithToyRoCC ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)
