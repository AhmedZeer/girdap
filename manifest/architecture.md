# Architecture

Atik is a small RoCC-attached tensor accelerator for matmul and online
attention. It follows basic ASIC-friendly principles: explicit DMA, local SRAM scratchpads, and a shared compute mesh.

## Top-Level Concept

```text
RoCC command
  -> descriptor reader
  -> operation controller
  -> simple DMA reader/writer
  -> hardware packers
  -> SRAM tile buffers
  -> BF16-to-fixed converters
  -> shared fixed-point MAC mesh
  -> accumulator/output path
  -> fixed-to-BF16 converters
  -> DMA writeback
```

For attention, the same path is extended with:

```text
score scaling
causal masking
online row max/sum update
exp LUT
reciprocal LUT
probability x V accumulation
```

## Shared Mesh Principle

Matmul, QK, and PV should all use the same MAC mesh.

```text
Matmul:
  C += A * B

Attention QK:
  score += Q * K^T

Attention PV:
  O += probability * V
```

This is the main difference from Girdap (older version, see branches), where matmul and attention are separate
RoCC accelerators.

## Memory Principle

Software passes row-major BF16 tensors. Hardware owns the transformation into
mesh-friendly vectors.

Large local storage should be implemented as SRAM-backed buffers. Small live
state, such as the active accumulator tile, row max, and row denominator, may
be registers. Local K chunk depth `KT` is hardware-configurable and may differ
from mesh size; it is not exposed through the software ABI.

## DMA Principle

The DMA should be a small tile DMA, not a full runtime system.

It should support:

- base address
- row count
- column count
- stride
- local SRAM destination/source
- read or write direction

The DMA exists to decouple memory movement from compute control. Operation
controllers request tile loads and stores; the DMA performs the memory traffic.

## Numeric Principle

External data:

```text
BF16 input
BF16 output
```

Internal data:

```text
BF16 -> signed fixed-point -> fixed-point accumulation -> BF16
```

For attention:

```text
scores are fixed-point
exp outputs are unsigned fixed-point
row sums are unsigned fixed-point
probability x V accumulation is signed fixed-point
```

Exponentiation and reciprocal may be lookup-table based.
