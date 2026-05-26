# Hidden Diagram Sources

This directory contains source assets for architecture diagrams that are not yet
published from the top-level README.

## Files

- `atik-architecture.svg`: editable SVG block diagram for the current Atik
  architecture.
- `atik-architecture.mmd`: Mermaid source for the same architecture hierarchy.
- `atik-architecture.html`: standalone HTML/CSS version of the paper-style
  architecture diagram.

## Hierarchy

The diagram intentionally shows architectural hierarchy, not every Chisel wire.
The goal is to explain the system in the same spirit as a paper or README
architecture figure: clear ownership, data movement, and compute reuse.

```text
Rocket / Host
  -> AtikRoCC
     -> AtikCommandRouter
     -> TileLink adapter
  -> AtikCore
     -> control
     -> descriptor DMA
     -> matmul / attention controllers
     -> shared tile DMA
     -> local tile storage
     -> shared MAC mesh
     -> attention scalar region
```

### Level 0: System Boundary

The outermost split is between the host system and the accelerator.

```text
Rocket / Host
  -> RoCC command path
  -> TileLink memory path
  -> main memory

Atik
  -> RoCC wrapper
  -> accelerator core
```

This makes the most important boundary explicit: Atik does not own the CPU or
DRAM. It receives commands through RoCC and moves tensor data through memory
requests that are eventually translated into TileLink transactions.

### Level 1: AtikRoCC Wrapper

`AtikRoCC` is shown as the external protocol wrapper. It should contain:

- `AtikCommandRouter`
- RoCC response path
- TileLink adapter for internal DMA requests
- `AtikTop` as the entry point into the accelerator core

The router is placed near the RoCC command input because it decodes software
commands such as `set_desc`, `run`, `status`, and counter reads. The TileLink
adapter is placed near the memory path because it turns Atik's internal
beat-level read/write requests into TileLink traffic.

### Level 2: AtikCore

`AtikCore` is the main accelerator box. It owns the implementation hierarchy:

- `AtikController`
- `DescriptorReader`
- descriptor `DmaReader`
- `MatmulController`
- `AttentionController`
- shared `TileDmaReader`
- shared `TileDmaWriter`
- local tile storage
- shared `MacMesh`
- counters and status registers

The diagram groups these into larger architectural regions instead of drawing
every module as a top-level box. This keeps the figure readable while still
matching the Chisel source structure.

### Level 3: Control Region

The control region contains `AtikController` and `DescriptorReader`.

`AtikController` stores the descriptor address, starts the descriptor fetch,
checks the decoded operation, and launches either matmul or attention. It is
drawn above the operation schedulers because it owns operation-level sequencing.

`DescriptorReader` is drawn inside control, but connected to DMA, because it is
logically part of the command/setup path while physically fetching the
software-visible `atik_desc_t` from memory.

### Level 4: Operation Schedulers

`MatmulController` and `AttentionController` are shown side by side because
they are mutually exclusive users of the shared datapath. They both request tile
loads and stores, drive the shared mesh, and report completion back to
`AtikController`.

The figure should make this design decision visible: Atik is not two separate
accelerators. It is one accelerator with two operation schedulers sharing memory
movement and compute resources.

### Level 5: DMA And Tile State

The DMA region shows two different roles:

- descriptor DMA: small beat-count fetch used for `atik_desc_t`
- tile DMA: row-major BF16 tile reads and writes for tensors

The tile state region represents SRAM-backed and register-backed local storage:

- A/Q tile SRAMs
- B/K tile SRAMs
- V tile state
- output tile state
- accumulator or score state owned by the controllers

This is grouped as "Tile State" instead of drawing every SRAM instance because
the important architectural point is locality: software passes row-major BF16
tensors, and hardware performs tiling, packing, buffering, and writeback.

### Level 6: Shared Compute Mesh

`MacMesh` is the central compute block. It should be visually large and shared
by both operation schedulers.

The diagram should communicate three uses of the same mesh:

```text
Matmul:
  C += A * B

Attention QK:
  score += Q * K^T

Attention PV:
  O += probability * V
```

This shared-mesh decision is one of the main architectural differences from the
older `girdap` branch, where matmul and attention paths were more separate.

### Level 7: Attention Scalar Region

Attention needs extra scalar work around the mesh:

- score scaling
- causal masking
- online row max/sum update
- exp LUT
- reciprocal LUT
- output normalization

This region is drawn near `AttentionController`, not as a second large compute
mesh. That reflects the current area-first design: QK and PV use the shared
mesh, while softmax and normalization are scalar-scheduled by default.

## Design Decisions

### Show Architecture, Not Netlist

Do not draw every Chisel signal. The diagram should explain how the accelerator
works at the subsystem level. Internal `Decoupled` handshakes, ready/valid
signals, muxes, and ownership registers are intentionally omitted unless they
explain an important architectural boundary.

### Keep RoCC And TileLink Separate

RoCC is the command path. TileLink is the memory path. The diagram keeps them
separate so readers do not confuse software commands with tensor data movement.

```text
RoCC:
  set_desc, run, status, counters

TileLink:
  descriptor fetch, tile loads, tile writeback
```

### Emphasize Hardware-Owned Tiling

Software only passes normal row-major BF16 tensors and a descriptor. Hardware
owns tile extraction, local storage layout, mesh feeding, and output packing.
This is why DMA and tile state are placed between controllers and compute.

### Emphasize Shared Resources

The figure should make these shared resources obvious:

- one shared `MacMesh`
- one shared tile read path
- one shared tile write path
- common status/counter infrastructure

Matmul and attention should appear as schedulers over shared hardware, not as
fully independent accelerator blocks.

### Keep Attention Area-First

Do not draw a wide softmax datapath unless the implementation changes. The
current default has one scalar lane for attention bookkeeping. The diagram
should show softmax as a scalar region around the shared mesh.

### Prefer Stable Names

Use public or source-level names that are stable:

- `AtikRoCC`
- `AtikTop`
- `AtikCore`
- `AtikController`
- `DescriptorReader`
- `MatmulController`
- `AttentionController`
- `TileDmaReader`
- `TileDmaWriter`
- `MacMesh`

Avoid temporary implementation names or small helper modules unless the diagram
is specifically about that subsystem.

## Publishing

When publishing, copy or export the SVG into `docs/static/` and reference it
from `README.md`.

To render the Mermaid source with Mermaid CLI:

```bash
mmdc -i .diagrams/atik-architecture.mmd -o .diagrams/atik-architecture-mermaid.svg
```
