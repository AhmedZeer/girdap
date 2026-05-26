# Scenario: Atik Attention

## Flow
To launch an attention kernel on Atik, the user first describes the operation: query shape, key/value shape, leading dimensions, tensor pointers, and the BF16 scale factor.
For that, we use the [atik_desc_t](../../software/include/atik_types.h) struct, which is the same struct used for [matmul](./matmul.md).
The user allocates the struct, fills in the relevant fields, and leaves unrelated parameters untouched. Calling [atik_rocc_set_desc()](../../software/include/atik_rocc.h) with a pointer to that struct emits the RoCC instruction that gives Atik the descriptor address.

After Atik has been configured with the chosen `atik_desc_t`, the user triggers execution by calling [atik_rocc_run()](../../software/include/atik_rocc.h), which emits the corresponding RoCC instruction. After that, the user waits until Atik finishes the work.

On the hardware side, the command bits arrive at the RoCC `io.cmd` interface. That interface connects to [AtikCommandRouter](../../src/main/scala/atik/rocc/AtikCommandRouter.scala), which decodes Atik instructions and generates internal control signals such as `setDesc` and `run`.

For the descriptor setup phase, the router raises `setDesc` and forwards the descriptor pointer through `descAddr`. [AtikTop](../../src/main/scala/atik/top/AtikTop.scala) simply passes those signals through to [AtikController](../../src/main/scala/atik/control/AtikController.scala). The controller stores the descriptor address and later starts [DescriptorReader](../../src/main/scala/atik/control/DescriptorReader.scala) when it sees `run`.

`DescriptorReader` then asks [DmaReader](../../src/main/scala/atik/memory/DmaReader.scala), which is wired as `descriptorDma` in [AtikCore](../../src/main/scala/atik/top/AtikCore.scala), to read `params.descriptorBeats` beats starting at the stored descriptor address. Once the beats are returned and parsed into `io.desc`, the reader raises `done`. At that point, `AtikController` inspects the descriptor, sees that the operation is attention or causal attention, records the causal flag when needed, and moves from `sReadDesc` to the attention launch path.

Next, `AtikController` raises `attentionStart` and forwards the decoded descriptor and causal mode into [AttentionController](../../src/main/scala/atik/control/AttentionController.scala). The attention controller checks the dimensions and base addresses, then enters its tiled schedule. It walks the query matrix in `MR` rows and the output value space in `NC` columns.

For each query/value tile, `AttentionController` initializes the running row maximum, row sum, and output accumulator. It then walks the key/value axis in chunks. For each chunk, it first loads Q and K through the shared tile DMA path in [AtikCore](../../src/main/scala/atik/top/AtikCore.scala), stores the returned BF16 elements into local SRAM-backed tile buffers, and reads them back in `KT` steps. Those values are converted to fixed-point and fed into the shared [MacMesh](../../src/main/scala/atik/compute/MacMesh.scala) to build the QK score tile.

After the score tile is computed, the controller applies the BF16 scale factor, masks future positions when causal attention is enabled, and runs the online softmax update with scalar scheduling. It computes the updated row maximum, rescales the previous output accumulator when needed, and stores `exp(score - max)` into local score-exp tile state. Then it loads the V tile through the shared tile DMA reader and folds `probability x V` through the shared mesh into the output accumulator.

When all key/value chunks for a query/value tile have been processed, the controller normalizes the accumulated output using the reciprocal of the final row sum, converts the result back to BF16, and writes the tile back through the shared tile DMA writer. It then moves to the next value tile or the next query tile until the full output matrix is complete.

When the last tile has been written, `AttentionController` raises `done`. `AtikController` observes that signal, marks the operation complete, and returns to the idle/done flow. Software typically polls status until `busy` clears and `done` is reported, then reads the final status word or continues with the next command.

## Goal

```text
O = softmax(Q * K^T * scale) * V
```

## External Format

Software passes normal row-major BF16 tensors:

```text
Q: q_rows x d_k, BF16
K: kv_rows x d_k, BF16
V: kv_rows x value_cols, BF16
O: q_rows x value_cols, BF16
```

## Internal State

For one query tile and one value-column tile:

```text
row_max:
  Vec<MR,SFixed<ACC_W,ACC_F>>

row_sum:
  Vec<MR,UFixed<SOFT_W,SOFT_F>>

out_acc:
  Tile<MR,NC,SFixed<ACC_W,ACC_F>>
```

The preferred design should not materialize the full attention score matrix in
memory. It should stream score tiles through online softmax and immediately
fold them into the output accumulation.

## Schedule

```text
for q0 in 0 .. q_rows step MR:
  for vcol0 in 0 .. value_cols step NC:

    row_max = -inf
    row_sum = 0
    out_acc = 0

    for kv0 in 0 .. kv_rows step KC:

      compute QK score tile:
        for d0 in 0 .. d_k step KT:
          load Q[q0 : q0+MR, d0 : d0+KT]
          load K[kv0 : kv0+KC, d0 : d0+KT]
          use shared mesh to accumulate score_tile += Q * K^T

      scale score tile:
        score_tile *= scale_bf16

      apply causal mask if requested:
        if global_k > global_q:
          score = NEG_INF

      update online softmax using scalar_lanes:
        tile_max = row max of score_tile
        new_max = max(row_max, tile_max)
        old_scale = exp(row_max - new_max)
        rescale out_acc one scheduled element at a time
        score_exp = exp(score_tile - new_max), computed into local tile state
        new_sum = row_sum * old_scale + sum(score_exp)

      load V tile:
        V[kv0 : kv0+KC, vcol0 : vcol0+NC]

      accumulate output:
        out_acc = out_acc * old_scale
        out_acc += score_exp * V

      row_max = new_max
      row_sum = new_sum

    normalize using scalar_lanes:
      inv_sum = reciprocal(row_sum)
      O_fixed = out_acc * inv_sum
      convert O_fixed to BF16 into the output tile buffer

    write O tile back to row-major memory
```


## Area-First Scalar Scheduling

The shared MAC mesh remains responsible for QK and PV work. The controller stores a local score_exp tile so PV can run as mesh outer-product steps instead of scalar probability x V multiplies. Online-softmax bookkeeping is a scalar scheduled region by default:

```text
attention_rows = MR
attention_kv_lanes = KC
attention_value_lanes = NC
attention_scalar_lanes = 1
exp_lut_instances = 1
reciprocal_lut_instances = 1
output_convert_lanes = 1
```

This trades extra cycles in softmax/update/normalization for a much smaller and
more placeable controller. Wider scalar lanes are an implementation knob, not an
ABI feature.

## Type Morphing

```text
Q/K/V from memory:
  BF16

Q/K/V mesh inputs:
  SFixed<IN_W,IN_F>

QK scores:
  SFixed<ACC_W,ACC_F>

exp(score - max):
  UFixed<SOFT_W,SOFT_F>

row_sum:
  UFixed<SOFT_W,SOFT_F>

probability x V output accumulator:
  SFixed<ACC_W,ACC_F>

final O:
  BF16
```

## Causality

Causal attention masks invalid future scores before softmax:

```text
global_q = q0 + local_q_row
global_k = kv0 + local_k_col

if global_k > global_q:
  score = NEG_INF
```

Masked scores do not affect row max, row sum, or PV accumulation.
