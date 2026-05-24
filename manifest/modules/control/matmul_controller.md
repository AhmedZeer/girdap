# Module: MatmulController

```yaml
module: MatmulController

purpose:
  Own the matmul tile loops and coordinate DMA, packers, SRAM, mesh compute,
  accumulator clear, and output writeback.

parameters:
  MR: mesh rows
  NC: mesh cols
  KT: internal K chunk depth; hardware configuration only, not software ABI

inputs:
  desc:
    type: AtikDescriptor
    meaning: matmul descriptor

  start:
    type: Bool
    meaning: begin matmul operation

outputs:
  a_read_request:
    type: Dma2DReadRequest
    meaning: request A tile chunk

  b_read_request:
    type: Dma2DReadRequest
    meaning: request B tile chunk

  c_write_request:
    type: Dma2DWriteRequest
    meaning: request C tile writeback

  clear_accumulator:
    type: Bool
    meaning: clear active output accumulator tile

  compute_chunk:
    type: ComputeChunkCommand
    meaning: compute loaded A/B SRAM chunk

  done:
    type: Bool
    meaning: matmul completed

state:
  m0:
    type: U32
  n0:
    type: U32
  k0:
    type: U32
    meaning: base K index of the active KT chunk

behavior: |
  for m0 in 0 .. M step MR:
    for n0 in 0 .. N step NC:
      clear_accumulator = true

      for k0 in 0 .. K step KT:
        active_KT = min(KT, K - k0)
        issue A read request
        wait for A load done

        issue B read request
        wait for B load done

        issue compute chunk command
        wait for compute done

      issue C write request
      wait for write done

  done = true
```
