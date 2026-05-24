# Module: TileSRAM

```yaml
module: TileSRAM

purpose:
  Store packed BF16 tile vectors close to the mesh.

parameters:
  DEPTH: number of vectors
  LANES: number of BF16 lanes per vector

matmul_binding:
  A_tile_sram_depth: KT
  A_tile_sram_lanes: MR
  B_tile_sram_depth: KT
  B_tile_sram_lanes: NC
  note: KT is an internal hardware configuration parameter and is not part of the ABI.

attention_binding:
  Q_tile_sram_depth: KT
  Q_tile_sram_lanes: MR
  K_tile_sram_depth: KT
  K_tile_sram_lanes: KC
  note: Attention uses the same internal KT hardware parameter for d_k chunks.

storage:
  mem:
    type: SRAM<DEPTH,Vec<LANES,BF16>>

inputs:
  write_index:
    type: U<log2(DEPTH)>

  write_data:
    type: Vec<LANES,BF16>

  write_enable:
    type: Bool

  read_index:
    type: U<log2(DEPTH)>

outputs:
  read_data:
    type: Vec<LANES,BF16>

behavior: |
  On load:
    if write_enable:
      mem[write_index] = write_data

  On compute:
    read_data = mem[read_index]
```

## ASIC Requirement

This block should map to SRAM macros or SRAM compiler output during VLSI flow.
It should not become a large flip-flop array.
