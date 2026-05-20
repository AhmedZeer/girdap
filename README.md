Girdap
=======================

Girdap is a Chipyard/RoCC accelerator project for BF16 matmul, online
attention, and softmax experiments.

### Current Hardware Configs

The active Chipyard configs are defined in `chipyard/GirdapConfigs.scala`:

| Flavor | Chipyard config | Accelerators |
| --- | --- | --- |
| Girdap İri | `Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512Config` | 8x8 BF16 FPGA-safe matmul plus 8x8 BF16 FPGA-safe online attention, hardware packer, exp LUT, `maxK=512` |
| Girdap Vasat | `FusedOnlineAttention8x8BF16FpgaSafePackerExpLutConfig` | 8x8 BF16 FPGA-safe online attention, hardware packer, exp LUT |
| Girdap Küçük | `MatmulAccel8x8BF16FpgaSafeConfig` | 8x8 BF16 FPGA-safe matmul |
| Girdap Minik | `SoftmaxAccel128Config` | standalone BF16 online softmax over the 128-bit system bus |

### Benchmarking Model

Girdap uses two benchmark families:

* Task-specific bare-metal C benchmarks generate their input data in C and
  measure one operation family directly: matmul, row softmax, or fused online
  attention. These are the best view of the raw datapath behavior for each
  hardware flavor.
* PyTorch-generated benchmarks export model and operator cases into C and run
  the same cases across hardware modes. These are the best view of end-to-end
  behavior after packing, memory traffic, software fallbacks, and non-accelerated
  model layers are included.

All speedups below are reported against the RocketCore software baseline from
the same benchmark (`sw_cycles / hw_cycles`). Passing rows have zero mismatches
unless otherwise noted.

### Task-Specific Performance

| Operation | Result source | Passing cases | Speedup vs RocketCore | Aggregate speedup | What it shows |
| --- | --- | ---: | ---: | ---: | --- |
| BF16 row softmax | `software/test_results/online_softmax_bf16inout_128bit.txt` | 5/5 | 116.8x-154.6x | 150.9x | The standalone online-softmax datapath is very efficient when the benchmark is only softmax over one row of 32-512 columns. |
| BF16 8x8 matmul | `software/test_results/safe_fpga_8x8_gemm_random.txt` | 10/10 | 3.6x-56.6x | 42.2x | Matmul improves with problem size: small 8x8x100 cases are dominated by packing and setup, while 256x256x256 reaches 56.6x. |
| BF16 8x8 online attention with HW packer | `software/test_results/hwpacker_onlineattn.txt` | 11/11 | 12.0x-102.1x | 93.8x | Fused attention benefits most when QK, online softmax, and PV stay inside one accelerator invocation; large 256-token cases are near 100x. |

The task-specific results are intentionally operation-centric. They show that
each accelerator can outperform RocketCore substantially in isolation, but they
do not predict full-model speedup by themselves because real models also include
projection layers, feed-forward layers, packing, copies, and control overheads.

### PyTorch-Generated Benchmarks

The archived PyTorch benchmark logs are under
`software/uartlog/Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512/`.
The collated per-case metrics are in
`software/uartlog/Matmul8x8AndOnlineAttention8x8BF16FpgaSafePackerExpLut512/metrics.csv`.
The table reports unique result rows; repeated TinyBERT attention+matmul rows in
`tinybert_attn_matmul_v2.txt` match rows already present in `v1`.

| Workload | Hardware mode | Passing cases | Speedup range | Geomean speedup | Aggregate speedup | Max abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Attention block/operator | attention+matmul | 28/28 | 12.0x-92.3x | 28.2x | 73.4x | 0.07507 |
| Attention block/operator | attention-only | 28/28 | 11.7x-92.3x | 28.2x | 73.4x | 0.07507 |
| Attention block/operator | matmul-only | 28/28 | 1.4x-13.5x | 3.2x | 9.3x | 0.09136 |
| Attention block/operator | softmax-only | 28/28 | 0.7x-1.9x | 1.1x | 0.8x | 0.16992 |
| TinyBERT | attention+matmul | 11/11 | 8.4x-71.8x | 27.4x | 56.5x | 0.03393 |
| TinyBERT | attention-only | 6/6 | 1.2x-1.8x | 1.4x | 1.5x | 0.00292 |
| TinyBERT | matmul-only | 5/5 | 2.8x-5.5x | 4.0x | 3.2x | 0.03222 |
| TinyBERT | softmax-only | 6/6 | 1.0x-1.1x | 1.0x | 1.0x | 0.00292 |
| GPT-2 prefill | attention+matmul | 6/6 | 6.2x-30.1x | 13.1x | 21.3x | 0.24218 |
| GPT-2 prefill | attention-only | 6/6 | 1.0x-1.1x | 1.0x | 1.1x | 0.00781 |
| GPT-2 prefill | matmul-only | 6/6 | 5.0x-13.5x | 7.5x | 9.3x | 0.24218 |
| GPT-2 prefill | softmax-only | 6/6 | 1.0x-1.0x | 1.0x | 1.0x | 0.04345 |

The main trend is that Girdap İri is the only configuration that consistently
turns the raw accelerator speed into model-level speedup. On TinyBERT it reaches
56.5x aggregate speedup, and on GPT-2 prefill it reaches 21.3x aggregate
speedup. The combined design does better than matmul-only because transformer
workloads spend meaningful time in both dense layers and attention, and it does
better than attention-only because the feed-forward and projection matmuls remain
large parts of the model.

Girdap Vasat is very strong on the attention block/operator benchmark because
the fused online-attention accelerator covers the whole QK-softmax-PV pattern in
one path. For full TinyBERT and GPT-2 prefill, however, attention-only speedup
is modest because the dense layers still run on RocketCore.

Girdap Küçük is the useful middle point for models with heavy dense compute. It
does not match the combined configuration, but it gives clear model-level gains:
3.2x aggregate on TinyBERT and 9.3x aggregate on GPT-2 prefill.

Girdap Minik is excellent as a standalone softmax accelerator, but softmax-only
model runs are near break-even. In the PyTorch workloads, accelerating only the
row softmax leaves QK, PV, projections, and feed-forward layers on RocketCore
and adds accelerator interface overhead. This is why the same softmax datapath
can show 150.9x aggregate speedup in the task-specific test while contributing
about 1x end-to-end speedup in full models.

### Licensing

Girdap is licensed under the Apache License, Version 2.0. See `LICENSE` and
`NOTICE`.
