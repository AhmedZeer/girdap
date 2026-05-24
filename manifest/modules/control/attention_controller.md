# Module: AttentionController

```yaml
module: AttentionController

purpose:
  Own attention tile loops and coordinate QK score computation, online softmax,
  V loading, probability x V accumulation, normalization, and output writeback.

parameters:
  MR: query rows per tile
  NC: value columns per output tile
  KT: d_k chunk depth
  KC: key/value rows per score tile

inputs:
  desc:
    type: AtikDescriptor
    meaning: attention descriptor

  start:
    type: Bool
    meaning: begin attention operation

outputs:
  q_read_request:
    type: Dma2DReadRequest

  k_read_request:
    type: Dma2DReadRequest

  v_read_request:
    type: Dma2DReadRequest

  o_write_request:
    type: Dma2DWriteRequest

  score_compute:
    type: ComputeChunkCommand

  softmax_update:
    type: SoftmaxUpdateCommand

  pv_accumulate:
    type: PvAccumulateCommand

  done:
    type: Bool

state:
  q0:
    type: U32
  vcol0:
    type: U32
  kv0:
    type: U32
  d0:
    type: U32

behavior: |
  for q0 in 0 .. q_rows step MR:
    for vcol0 in 0 .. value_cols step NC:
      initialize row_max, row_sum, and out_acc

      for kv0 in 0 .. kv_rows step KC:
        clear score tile

        for d0 in 0 .. d_k step KT:
          load Q chunk through DmaReader
          load K chunk through DmaReader
          accumulate QK score tile on shared mesh

        scale scores
        apply causal mask if requested
        update online softmax row state

        load V tile through DmaReader
        accumulate probability x V into out_acc on shared mesh

      normalize out_acc by row_sum
      write output tile

  done = true
```

## Implementation Notes

```yaml
q_k_loads:
  storage: SRAM-backed Q and K tile buffers, depth KT
  reader: shared beat-aware TileDmaReader
  schedule: load Q[MR, active_KT] and K[KC, active_KT], then iterate active_KT
    locally into the shared mesh
  reason: make the attention d_k chunk match the manifest KT model and avoid
    one DMA command per feature element

v_loads:
  storage: local tile registers for first implementation
  reader: shared beat-aware TileDmaReader
  reason: V is consumed once by the PV stage after online-softmax update

output_write:
  writer: shared TileDmaWriter
  schedule: write the normalized MR x NC output tile after online softmax state is
    complete for the current query/value tile
  reason: share one packed row-major BF16 writeback implementation with matmul
```
