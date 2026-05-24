# Scenario: Atik Matmul

## Goal

```text
C[M,N] = A[M,K] * B[K,N]
```

## External Format

Software passes normal row-major BF16 tensors:

```text
A: M x K, BF16
B: K x N, BF16
C: M x N, BF16
```

Software does not pack tiles.

## Internal Format

```text
A SRAM:
  A_sram[kk] = Vec<MR,BF16>

B SRAM:
  B_sram[kk] = Vec<NC,BF16>

Mesh input:
  Vec<MR,SFixed<IN_W,IN_F>>
  Vec<NC,SFixed<IN_W,IN_F>>

Accumulator:
  Tile<MR,NC,SFixed<ACC_W,ACC_F>>

Output:
  Tile<MR,NC,BF16>
```

## Schedule

`KT` is an internal hardware configuration parameter. It is not visible in
the software ABI. The descriptor `K` may be larger than `KT`; hardware keeps
the accumulator live across all K chunks for an output tile.

```text
for each output tile C[m0 : m0+MR, n0 : n0+NC]:

  clear accumulator tile

  for k0 in 0 .. K step KT:

    load A chunk:
      read A[m0 : m0+MR, k0 : k0+KT]
      row-major BF16 memory -> packed A vectors
      store into A_sram[0 .. active_KT-1]

    load B chunk:
      read B[k0 : k0+KT, n0 : n0+NC]
      row-major BF16 memory -> packed B vectors
      store into B_sram[0 .. active_KT-1]

    compute chunk:
      for kk in 0 .. active_KT-1:
        A_vec_bf16 = A_sram[kk]
        B_vec_bf16 = B_sram[kk]

        A_vec_fixed = bf16_to_fixed(A_vec_bf16)
        B_vec_fixed = bf16_to_fixed(B_vec_bf16)

        accumulator += outer_product(A_vec_fixed, B_vec_fixed)

  convert accumulator tile to BF16

  write C tile back to row-major memory
```

## Edge Tiles

If the tile crosses matrix bounds:

```text
invalid A lanes become BF16 zero
invalid B lanes become BF16 zero
invalid C lanes are not written
```

## Key Point

There is no architectural `maxK` for the full problem. The full descriptor `K`
may be larger than local storage. Hardware repeatedly loads K chunks of depth
`KT` and keeps the accumulator live until all chunks are complete. `KT` may be
larger than the mesh dimension when SRAM capacity allows it, but changing it is
a hardware configuration choice, not an ABI change.
