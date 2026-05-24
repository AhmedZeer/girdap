# Counters

Counters are for performance analysis and debug. They are not required for
correctness.

## Useful Counters

```text
total_cycles
  Cycles from accepted RUN to done.

dma_read_cycles
  Cycles with active read DMA.

dma_write_cycles
  Cycles with active write DMA.

compute_cycles
  Cycles spent issuing mesh compute steps.

mesh_active_cycles
  Cycles where the mesh performs useful work.

mesh_idle_cycles
  Cycles where an operation is active but the mesh is waiting.

softmax_cycles
  Attention-only cycles spent in softmax/update/normalization.

bytes_read
  Total bytes read by DMA.

bytes_written
  Total bytes written by DMA.

tiles_loaded
  Number of input tile chunks loaded.

tiles_computed
  Number of compute chunks completed.

dma_stall_cycles
  Cycles blocked by memory/DMA availability.

sram_stall_cycles
  Cycles blocked by local SRAM access conflicts.
```

## Implemented Counter Set

The first eight counter indices are stable and preserve the original minimum set:

```text
total_cycles
compute_cycles
dma_read_cycles
dma_write_cycles
mesh_active_cycles
mesh_idle_cycles
bytes_read
bytes_written
```

Additional manifest counters are appended after the original set:

```text
softmax_cycles
tiles_loaded
tiles_computed
dma_stall_cycles
sram_stall_cycles
```

These answer the most important early questions:

- Is the design compute-bound or memory-bound?
- Is DMA keeping the mesh fed?
- Does scaling from 2x2 to 4x4 to 8x8 behave as expected?
- Is attention dominated by softmax or by QK/PV compute?
