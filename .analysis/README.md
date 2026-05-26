# UART Benchmark Analysis

Generated from `software/uartlog/*/RESULT_JSON` records.

## Inputs

- `2x2`: `software/uartlog/atik_2x2`
- `4x4`: `software/uartlog/atik_4x4`
- `8x8`: `software/uartlog/atik_8x8_v2`

The older `atik_8x8_v1` logs are not used because `atik_8x8_v2` contains the fuller matching workload set.

## Figures

- `figures/*_speedup.svg`: per-case speedup lines. The x-axis is problem size, and the y-axis is speedup over RocketCore.
- `figures/aggregate_speedup.svg`: aggregate speedup by summing CPU cycles and hardware cycles over common problem shapes in each workload.
- `figures/aggregate_cycles.svg`: aggregate cycle bars with speedup annotations from RocketCore to each accelerator config.
- `figures/peak_speedup.svg`: best single-case speedup observed for each workload/config across all logged cases.
- `figures/peak_cycles.svg`: matched RocketCore and accelerator clock cycles for each config's own best-speedup case.

RocketCore is shown as a grey 1x baseline. Atik 2x2, 4x4, and 8x8 use progressively darker purple tones.

## Aggregate Speedups

| Workload | 2x2 | 4x4 | 8x8 |
|---|---:|---:|---:|
| Matmul Benchmark | 4.03x | 11.90x | 30.97x |
| Attention Benchmark | 9.35x | 43.71x | 181.71x |
| ViT Workload | 8.02x | 22.84x | 44.18x |
| TinyBERT Workload | 7.35x | 19.78x | 36.10x |
| GPT-2 Prefill Workload | 5.57x | 15.30x | 32.31x |

## Notes

- Speedup is computed as `cpu_cycles / hw_cycles`.
- Aggregate speedup is computed as `sum(cpu_cycles) / sum(hw_cycles)` over problem shapes present in all three configs.
- Only records with `status == PASS` are included in plots and aggregate calculations.
