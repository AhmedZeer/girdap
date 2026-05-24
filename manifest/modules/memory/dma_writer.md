# Module: DmaWriter

```yaml
module: DmaWriter

purpose:
  Write BF16 output tiles back to row-major memory.

inputs:
  request:
    type: Dma2DWriteRequest
    meaning: base address, rows, cols, stride, element size, edge masks

  data:
    type: Stream<BF16>
    meaning: output BF16 elements in row-major tile order

outputs:
  done:
    type: Bool

  error:
    type: AtikStatus

behavior: |
  For each output row:
    collect BF16 elements from the stream
    write valid columns to memory
    do not write invalid edge columns

  done = true after all valid output elements are written.
```


## Current Bindings

```yaml
matmul:
  writes: C tile
  layout: row-major BF16

attention:
  writes: O tile
  layout: row-major BF16
```
