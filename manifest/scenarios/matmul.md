# Scenario: Atik Matmul

## Flow
To launch a matmul kernel on Atik, the user first describes the operation: matrix dimensions, leading dimensions, and tensor pointers.
For that, we use the [atik_desc_t](../../software/include/atik_types.h) struct, which is the same struct used for [attention](./attention.md).
The user allocates the struct, fills in the relevant fields, and leaves unrelated parameters untouched. Calling [atik_rocc_set_desc()](../../software/include/atik_rocc.h) with a pointer to that struct emits the RoCC instruction that gives Atik the descriptor address.

After Atik has been configured with the chosen `atik_desc_t`, the user triggers execution by calling [atik_rocc_run()](../../software/include/atik_rocc.h), which emits the corresponding RoCC instruction. After that, the user waits until Atik finishes the work.

On the hardware side, the command bits arrive at the RoCC `io.cmd` interface. That interface connects to [AtikCommandRouter](../../src/main/scala/atik/rocc/AtikCommandRouter.scala), which decodes Atik instructions and generates internal control signals such as `setDesc` and `run`.

For the descriptor setup phase, the router raises `setDesc` and forwards the descriptor pointer through `descAddr`. [AtikTop](../../src/main/scala/atik/top/AtikTop.scala) simply passes those signals through to [AtikController](../../src/main/scala/atik/control/AtikController.scala). The controller stores the descriptor address and later starts [DescriptorReader](../../src/main/scala/atik/control/DescriptorReader.scala) when it sees `run`.

`DescriptorReader` then asks [DmaReader](../../src/main/scala/atik/memory/DmaReader.scala), which is wired as `descriptorDma` in [AtikCore](../../src/main/scala/atik/top/AtikCore.scala), to read `params.descriptorBeats` beats starting at the stored descriptor address. Once the beats are returned and parsed into `io.desc`, the reader raises `done`. At that point, `AtikController` inspects the descriptor, sees that the operation is matmul, and moves from `sReadDesc` to the matmul launch path.

Next, `AtikController` raises `matmulStart` and forwards the decoded descriptor into [MatmulController](../../src/main/scala/atik/control/MatmulController.scala). The matmul controller checks the dimensions and base addresses, then enters its tile loop. It walks the output matrix in `MR x NC` tiles, clears the active accumulator tile, and then processes the full `K` dimension in internal chunks of depth `KT`.

For each chunk, `MatmulController` issues two tile DMA reads through the shared tile DMA path in [AtikCore](../../src/main/scala/atik/top/AtikCore.scala): one for the A tile slice and one for the B tile slice. The tile DMA readers fetch row-major BF16 data from memory, and the controller stores the returned elements into local SRAM-backed tile buffers. Once both inputs are loaded, the controller reads them back from SRAM, converts BF16 values to fixed-point, and drives the shared [MacMesh](../../src/main/scala/atik/compute/MacMesh.scala) for the multiply-accumulate step.

That same accumulator stays live across all `KT` chunks belonging to the same output tile. When the full `K` range for the tile has been accumulated, the controller converts the result back to BF16 and issues a tile writeback through the shared tile DMA writer. It then moves to the next output tile and repeats the process until the full `M x N` output matrix is done.

When the last tile has been written, `MatmulController` raises `done`. `AtikController` observes that signal, marks the operation complete, and returns to the idle/done flow. Software typically polls status until `busy` clears and `done` is reported, then reads the final status word or continues with the next command.



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
