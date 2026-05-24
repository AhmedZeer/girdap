# Atik Chisel Source Map

This directory contains the modular Chisel implementation of Atik. The hardware is organized around a small RoCC-facing ABI, a shared fixed-point mesh, reusable DMA helpers, SRAM-backed tile buffers, and two operation controllers: matmul and online attention.

## Root Package

- `main/scala/atik/AtikParams.scala`
  Defines mesh size, BF16 width, fixed-point widths, accumulator width, softmax/LUT widths, address width, memory beat width, descriptor size, counter count, and internal `matmulKt`. This is the main knob for 2x2, 4x4, and 8x8 builds. `matmulKt` is also used as the attention `d_k` chunk depth.

- `main/scala/atik/AtikOpcodes.scala`
  Defines software-visible RoCC function codes, operation IDs, status/error codes, and counter indices. These mirror the ABI headers in `software/include`.

- `main/scala/atik/AtikTypes.scala`
  Defines shared Chisel bundles: the software descriptor layout, host command/response records, status state, counter events, and generic DMA beat request/response records.

## RoCC Boundary

- `main/scala/atik/rocc/AtikCommandRouter.scala`
  Decodes RoCC commands into core control pulses and formats RoCC responses for status and counter reads.

- `main/scala/atik/rocc/AtikRoCC.scala`
  Provides the LazyRoCC wrapper, TileLink client node, TileLink beat adapter for DMA requests, and config fragments for 2x2, 4x4, and 8x8 Atik instances.

## Top-Level Wiring

- `main/scala/atik/top/AtikCore.scala`
  Connects command control, descriptor DMA, matmul and attention controllers, shared MAC mesh, counters, status packing, and DMA arbitration. Matmul and attention do not run concurrently, so the shared mesh is selected by the active controller.

- `main/scala/atik/top/AtikTop.scala`
  Thin wrapper around `AtikCore`; intended as the stable top module for RTL emission and VLSI flows.

## Control

- `main/scala/atik/control/AtikController.scala`
  Main operation FSM. It stores descriptor address, starts descriptor DMA, validates operation type, launches matmul or attention, and drives busy/done/error state.

- `main/scala/atik/control/DescriptorReader.scala`
  Reads the 80-byte software descriptor through the DMA reader and parses it for 64-bit or 128-bit memory beats.

- `main/scala/atik/control/MatmulController.scala`
  Matmul tile scheduler. It walks M/N tiles, chunks K by internal `matmulKt`, loads A/B chunks through `TileDmaReader`, stores them in SRAM-backed tile buffers, reads one K lane per mesh step, accumulates on the shared mesh, converts the result tile to BF16, and writes it through `TileDmaWriter`.

- `main/scala/atik/control/AttentionController.scala`
  Online attention tile scheduler. It walks query, value-column, and key/value tiles. Q/K are loaded in `KT` chunks through `TileDmaReader` into SRAM-backed tile buffers, QK scores accumulate on the shared mesh, causal masking and online softmax state are updated locally, V tiles are read through `TileDmaReader`, PV accumulation uses the shared mesh, normalized output is converted to BF16, and the O tile is written through `TileDmaWriter`.

- `main/scala/atik/control/CounterBank.scala`
  Implements ABI-visible performance counters: total cycles, compute cycles, DMA read/write cycles, mesh active/idle cycles, bytes read/written, softmax cycles, tile load/compute events, DMA stalls, and SRAM stalls.

- `main/scala/atik/control/StatusRegs.scala`
  Packs busy/done/error into the ABI status word.

## Memory

- `main/scala/atik/memory/DmaReader.scala`
  Small beat-count DMA reader used by descriptor-style transfers. It issues memory read beats and streams returned data.

- `main/scala/atik/memory/DmaWriter.scala`
  Small beat-count DMA writer kept as a simple bulk-write primitive. Current matmul and attention tile writeback use `TileDmaWriter`.

- `main/scala/atik/memory/TileDma.scala`
  Contains reusable row-major tile DMA engines. `TileDmaReader` reads BF16 tensor regions, caches returned memory beats, and emits BF16 elements in row-major order. `TileDmaWriter` packs BF16 output tiles into aligned memory beats with byte masks.

- `main/scala/atik/memory/DmaRequestArbiter.scala`
  Round-robin arbiter for multiple DMA read command producers.

- `main/scala/atik/memory/SramBank.scala`
  Synthesizable `SyncReadMem` bank wrapper. This is the macro replacement point for VLSI SRAM mapping and the inference target for FPGA BRAM/LUTRAM.

- `main/scala/atik/memory/TileSram.scala`
  Tile-buffer wrapper around `SramBank`. Matmul uses it for A/B `KT` chunks; attention uses it for Q/K `KT` chunks.

- `main/scala/atik/memory/SramLayout.scala`
  Names local SRAM regions used by the datapath.

- `main/scala/atik/memory/InputPacker.scala`
  Legacy/simple BF16 input packing helper for mesh-side lane vectors. The current tiled controllers primarily use `TileDmaReader` plus tile SRAMs.

- `main/scala/atik/memory/OutputPacker.scala`
  Legacy/simple BF16 output packing helper. The current tiled controllers primarily use `TileDmaWriter`.

## Compute

- `main/scala/atik/compute/Bf16ToFixed.scala`
  Converts BF16 inputs into signed fixed-point values. Zero/subnormal values map to zero and large values clip.

- `main/scala/atik/compute/FixedToBf16.scala`
  Converts signed fixed-point/accumulator values back into BF16-like output encoding with basic underflow/overflow handling.

- `main/scala/atik/compute/MacCell.scala`
  One signed fixed-point multiply-accumulate cell.

- `main/scala/atik/compute/MacMesh.scala`
  Parameterized square mesh of MAC cells for 2x2, 4x4, or 8x8 builds. It is shared by matmul QK/PV attention paths.

- `main/scala/atik/compute/AccumulatorTile.scala`
  Register-backed accumulator tile helper. Current tiled controllers keep their operation-specific accumulator state locally.

## Attention Helpers

- `main/scala/atik/attention/CausalMask.scala`
  Applies causal masking by invalidating scores where `k_index > q_index`.

- `main/scala/atik/attention/ScoreScaler.scala`
  Applies the attention scale factor to QK scores in fixed-point.

- `main/scala/atik/attention/ExpLut.scala`
  Lookup-table approximation for `exp(x)` over negative fixed-point score deltas.

- `main/scala/atik/attention/ReciprocalLut.scala`
  Lookup-table reciprocal approximation for softmax normalization.

- `main/scala/atik/attention/OnlineSoftmax.scala`
  Online softmax row update helper: running max, denominator update, and current score exponent.

- `main/scala/atik/attention/ProbVAccumulator.scala`
  Fixed-point accumulation helper for `probability * V`. The current controller maps PV accumulation onto the shared mesh.

- `main/scala/atik/attention/AttentionNormalize.scala`
  Applies a reciprocal denominator to an accumulated attention output.

## Utility

- `main/scala/atik/util/FixedPointUtil.scala`
  Small helpers for fixed-point resizing, narrowing, and unsigned resizing.
