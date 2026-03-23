package chipyard

import org.chipsalliance.cde.config.Config

class DummyToyRoCCConfig extends Config(
  new toyrocc.WithToyRoCC ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)
