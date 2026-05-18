Girdap
=======================

Girdap is a Chipyard/RoCC accelerator project for BF16 matmul, online
attention, and softmax experiments.

### Current Hardware Configs

The active Chipyard configs are defined in `chipyard/GirdapConfigs.scala`:

* `FusedOnlineAttention8x8BF16FpgaSafePackerExpLutConfig`
* `Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLutConfig`
* `Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512Config`
* `SoftmaxAccel128Config`
* `MatmulAccel8x8BF16FpgaSafeConfig`

### Licensing

New BF16 matmul, attention, FPGA-safe accelerator, software workload, and
FireSim/Chipyard integration extensions added for this project are licensed
under Apache-2.0. See `LICENSE.Apache-2.0` and `NOTICE`.

Third-party code inherited from Chipyard, FireSim, Rocket Chip, Gemmini, and
other dependencies remains under its original upstream license.
