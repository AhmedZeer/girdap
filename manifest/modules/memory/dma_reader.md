# Module: DmaReader

```yaml
module: DmaReader

purpose:
  Read row-major BF16 tensor regions from memory and stream the values to a
  hardware packer.

inputs:
  request:
    type: Dma2DReadRequest
    meaning: base address, rows, cols, stride, element size

outputs:
  data:
    type: Stream<BF16>
    meaning: BF16 elements from the requested region

  done:
    type: Bool

  error:
    type: AtikStatus

behavior: |
  For each requested row-major element:
    compute the element address
    align the address down to the memory beat boundary
    if the current cached beat already contains the element:
      emit the BF16 lane from the cached beat
    else:
      issue one aligned memory read beat
      cache the returned beat
      emit the requested BF16 lane

  Continue draining adjacent BF16 lanes from the cached beat before issuing
  another request. Strided row transitions are supported; if the next row starts
  in the same aligned beat it may be emitted from the same cached beat.

  done = true after all elements are emitted.
```

## Notes

The first DMA is still blocking at the tile level: the controller loads the tile
before consuming it. The reader is beat-aware, so contiguous BF16 elements share
one memory request instead of one request per element. Later implementations may
overlap DMA and compute with double buffering.
