# Weight-Stationary Systolic Array Optimization Log

## Scope

These archived notes track earlier optimization work on the `4x4`
weight-stationary RoCC systolic array path. The active Girdap hardware config
surface has since been reduced to the configs listed in
`chipyard/GirdapConfigs.scala`.

- Archived config: `chipyard.MatmulAccel4x4WSConfig`
- Output-stationary path is intentionally ignored here

## Archived Config Facts

- The archived WS config used a `128-bit` system bus.
- A separate archived `256-bit` comparison config existed as `chipyard.MatmulAccel4x4WS256Config`.
- The WS RTL derives its transfer width from `cacheDataBits`, so wider TL beats directly increase `wordsPerBeat` in [src/main/scala/SystolicArrayWS.scala](/media/azeer/extra-segment/git/chipyard/generators/girdap/src/main/scala/SystolicArrayWS.scala).
- With `xLen = 64` and `systemBusWidth = 128`, the current design moves `2` packed words per beat.

## Iterations Completed

### 1. Hardware perf counters

Added internal WS counters for:

- accelerator busy cycles
- run and preload command counts
- preload reuse hits
- chunk count
- TL A/B/C transaction counts
- wait cycles for B fill, A fill, chunk drain, and C store

Files:

- [src/main/scala/SystolicArrayWS.scala](/media/azeer/extra-segment/git/chipyard/generators/girdap/src/main/scala/SystolicArrayWS.scala)
- [software/include/systolic_ws.h](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/include/systolic_ws.h)

### 2. Beat-buffered TL traffic

Changed the WS path from one-word TL accesses to beat-sized TL accesses for:

- A reads
- B reads
- C writes

Effect:

- TL transaction count dropped roughly `2x` on the current `128-bit` bus
- simple WS test improved from `7159` to `6124` cycles

### 3. A-side ping-pong prefetch

Added double-buffered A chunk storage so the next A chunk can be fetched while the current chunk computes.

Effect:

- simple WS test improved from `6124` to `5749` cycles

### 4. Software A-tile reuse

Changed the WS software drivers so A tiles are packed once per `m0` tile and reused across `n0` tiles, instead of being rebuilt repeatedly.

Files:

- [software/src/systolic_matmul_random_weight_stationary.c](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/src/systolic_matmul_random_weight_stationary.c)
- [software/src/systolic_matmul_benchmark.c](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/src/systolic_matmul_benchmark.c)

### 5. Fused `run_preloaded` command

Added a WS command which runs directly against already-preloaded weights, removing the per-tile `config + run` pair from the hot loop.

Effect on random WS:

- `4x4x100`: `8434 -> 8416`
- `8x8x100`: `25543 -> 18488`
- `16x16x100`: `86261 -> 45166`

Interpretation:

- this was a major control-path win for multi-tile cases
- the `4x4x100` case barely changed because it only has one output tile

### 6. Software-stage timing

Added software-side stage timing to split end-to-end cycles into:

- `pack_a_cycles`
- `pack_b_cycles`
- `preload_cycles`
- `run_cycles`
- `copy_out_cycles`

Also fixed software dependency generation so header changes rebuild binaries automatically.

Files:

- [software/src/systolic_matmul_random_weight_stationary.c](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/src/systolic_matmul_random_weight_stationary.c)
- [software/src/systolic_matmul_benchmark.c](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/src/systolic_matmul_benchmark.c)
- [software/Makefile](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/Makefile)

### 7. Shared WS GEMM library

Moved the weight-stationary GEMM path out of the test programs into a reusable library.

New files:

- [software/include/ws_gemm.h](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/include/ws_gemm.h)
- [software/common/ws_gemm.c](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/common/ws_gemm.c)

Current library entry points:

- `ws_gemm_u16(...)`
- `ws_gemm_u16_prepacked_b(...)`
- `ws_gemm_pack_b_u16(...)`

Design intent:

- `ws_gemm_u16(...)` is the generic row-major GEMM API for current users
- `ws_gemm_u16_prepacked_b(...)` exists for future weight reuse, which matters for transformer inference
- current WS tests and benchmarks should become thin wrappers over the library instead of owning the kernel logic directly

### 8. Separate `256-bit` WS config

Added a second WS config so bus-width experiments do not overwrite the `128-bit` baseline.

File:

- [chipyard/GirdapConfigs.scala](/media/azeer/extra-segment/git/chipyard/generators/girdap/chipyard/GirdapConfigs.scala)

Config:

- `chipyard.MatmulAccel4x4WS256Config`

Measured effect on random WS:

- `4x4x100`: `4724 -> 4015`
- `8x8x100`: `9884 -> 7877`
- `16x16x100`: `26556 -> 19605`

Measured hardware-counter effect on `16x16x100`:

- `busy_cycles`: `10463 -> 7849`
- `tl_b_reads`: `200 -> 100`
- `tl_a_reads`: `800 -> 400`
- `tl_c_writes`: `128 -> 64`
- `wait_fill_b_cycles`: `1680 -> 1072`
- `wait_fill_a_cycles`: `5303 -> 2902`
- `wait_put_cycles`: `960 -> 577`

Interpretation:

- widening the bus gave another strong real speedup
- the transfer-count hypothesis was correct
- the remaining easy RTL wins are now smaller and more structural

### 9. Pending RTL experiment: multi-outstanding B fills

Implemented and validated:

- multiple in-flight TL source IDs for B-tile preload in [src/main/scala/SystolicArrayWS.scala](/media/azeer/extra-segment/git/chipyard/generators/girdap/src/main/scala/SystolicArrayWS.scala)

Design intent:

- keep several B-cache beat reads in flight instead of strict request/response serialization
- target the remaining preload-side memory latency without changing the PE array or software API

Expected effect:

- `tl_b_reads` should stay the same
- `wait_fill_b_cycles` and `preload_cycles` should drop
- end-to-end gains should be most visible on larger `K`

Measured effect on the `256-bit` config:

- `4x4x100`: `4015 -> 3835`
- `8x8x100`: `7877 -> 7528`
- `16x16x100`: `19605 -> 18909`

Measured hardware-counter effect on `16x16x100`:

- `tl_b_reads`: unchanged at `100`
- `wait_fill_b_cycles`: `1072 -> 472`
- `preload_cycles`: `1223 -> 527`
- `busy_cycles`: `7849 -> 7153`

Interpretation:

- the change behaved as intended
- preload-side serialization is now much less exposed
- remaining hardware-side gains will require either deeper overlap on A/output traffic or a larger interface change

### 10. Prepared-weight API for model code

Added a cleaner software path for weight reuse in the shared library:

- `ws_gemm_prepare_packed_b_u16(...)` in [software/include/ws_gemm.h](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/include/ws_gemm.h)
- implementation in [software/common/ws_gemm.c](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/common/ws_gemm.c)

Also added a dedicated reuse benchmark:

- [software/src/systolic_matmul_reuse_weight_stationary.c](/media/azeer/extra-segment/git/chipyard/generators/girdap/software/src/systolic_matmul_reuse_weight_stationary.c)

Design intent:

- pack static weights once
- run multiple GEMMs against the same prepared weights
- measure the exact benefit of the model-facing reuse path separately from microkernel-only tuning

Measured effect from the reuse benchmark with repeated serial calls:

- `8x8x100`, `4` repeats: `31511 -> 24451` plus one-time pack `1545`
- `16x16x100`, `4` repeats: `78638 -> 64737` plus one-time pack `3119`

Measured stage effect:

- prepared serial runs removed repeated `pack_b_cycles`
- prepared serial runs still paid repeated `preload_cycles`, because each GEMM call reloaded the packed B tiles

Deferred finding:

- a batched prepared-weight API was tried and then reverted
- on the measured reuse cases it regressed badly because it repacked `A` per `N` tile, so it is not a good next step in the current software structure

## Latest Measured Results

### Simple WS

- after counters only: `7159`
- after beat-buffered traffic: `6124`
- after A prefetch overlap: `5749`

### Random WS

Before fused preloaded run:

- `4x4x100`: `8434`
- `8x8x100`: `25543`
- `16x16x100`: `86261`

After fused preloaded run:

- `4x4x100`: `8416`
- `8x8x100`: `18488`
- `16x16x100`: `45166`

After software fast-path packing/store:

- `4x4x100`: `5513`
- `8x8x100`: `12182`
- `16x16x100`: `30919`

Latest hardware counters for `16x16x100` after the fused run path:

- `busy_cycles = 10508`
- `run_cmds = 16`
- `preload_cmds = 4`
- `tl_b_reads = 200`
- `tl_a_reads = 800`
- `tl_c_writes = 128`
- `wait_fill_b_cycles = 1725`
- `wait_fill_a_cycles = 5302`
- `wait_chunk_out_cycles = 4834`
- `wait_put_cycles = 962`

Latest stage breakdown for `16x16x100` after the software fast paths:

- `pack_a_cycles = 7660`
- `pack_b_cycles = 6498`
- `preload_cycles = 1969`
- `run_cycles = 256`
- `copy_out_cycles = 1196`

Latest measured results with the shared WS kernel after the library fast-path fixes:

- `4x4x100`: `4724`
- `8x8x100`: `9884`
- `16x16x100`: `26556`

Latest stage breakdown for `16x16x100` with the shared WS kernel:

- `pack_a_cycles = 6383`
- `pack_b_cycles = 2591`
- `preload_cycles = 1931`
- `run_cycles = 8784`
- `copy_out_cycles = 1352`

Latest hardware counters for `16x16x100` with the shared WS kernel:

- `busy_cycles = 10463`
- `run_cmds = 16`
- `preload_cmds = 4`
- `tl_b_reads = 200`
- `tl_a_reads = 800`
- `tl_c_writes = 128`
- `wait_fill_b_cycles = 1680`
- `wait_fill_a_cycles = 5303`
- `wait_chunk_out_cycles = 4835`
- `wait_put_cycles = 960`

Latest measured results on the `256-bit` WS config:

- `4x4x100`: `4015`
- `8x8x100`: `7877`
- `16x16x100`: `19605`

Latest stage breakdown for `16x16x100` on the `256-bit` WS config:

- `pack_a_cycles = 6333`
- `pack_b_cycles = 2946`
- `preload_cycles = 1223`
- `run_cycles = 6856`
- `copy_out_cycles = 1349`

Latest hardware counters for `16x16x100` on the `256-bit` WS config:

- `busy_cycles = 7849`
- `run_cmds = 16`
- `preload_cmds = 4`
- `tl_b_reads = 100`
- `tl_a_reads = 400`
- `tl_c_writes = 64`
- `wait_fill_b_cycles = 1072`
- `wait_fill_a_cycles = 2902`
- `wait_chunk_out_cycles = 3488`
- `wait_put_cycles = 577`

Latest measured results on the `256-bit` config after multi-outstanding B fills:

- `4x4x100`: `3835`
- `8x8x100`: `7528`
- `16x16x100`: `18909`

Latest stage breakdown for `16x16x100` after multi-outstanding B fills:

- `pack_a_cycles = 6333`
- `pack_b_cycles = 2946`
- `preload_cycles = 527`
- `run_cycles = 6856`
- `copy_out_cycles = 1349`

Latest hardware counters for `16x16x100` after multi-outstanding B fills:

- `busy_cycles = 7153`
- `run_cmds = 16`
- `preload_cmds = 4`
- `tl_b_reads = 100`
- `tl_a_reads = 400`
- `tl_c_writes = 64`
- `wait_fill_b_cycles = 472`
- `wait_fill_a_cycles = 2902`
- `wait_chunk_out_cycles = 3488`
- `wait_put_cycles = 577`

Current interpretation:

- end-to-end `18909` cycles is still much larger than accelerator busy `7153`
- the next bottleneck is not the PE mesh
- remaining cost is now mostly software staging plus fixed per-chunk drain/output behavior
- software packing is still material, and the strongest remaining RTL-side signals are A-side fill exposure and `wait_chunk_out`
- future generic-kernel work should focus on reusable packing and weight reuse rather than benchmark-only loops

## PERF vs STAGE

### `PERF`

`PERF_*` lines come from hardware counters inside the accelerator RTL.

Use them to understand:

- what the accelerator itself is doing
- how many TL transactions occurred
- where the RTL waited

### `STAGE`

`STAGE_*` lines come from software-side `rdcycle` timing around major phases in the WS driver.

Use them to understand:

- how much time is spent packing matrices
- how much time is spent issuing preload/run commands
- how much time is spent copying results out

Short version:

- `PERF` explains accelerator-internal behavior
- `STAGE` explains software-visible end-to-end breakdown around the accelerator

## Can A Wider System Bus Help?

Yes, potentially.

If `systemBusWidth` in [chipyard/GirdapConfigs.scala](/media/azeer/extra-segment/git/chipyard/generators/girdap/chipyard/GirdapConfigs.scala) is increased from `128` to `256`, then with `xLen = 64` the WS RTL would move `4` packed words per beat instead of `2`.

What that should help:

- fewer A beat reads
- fewer B beat reads
- fewer C beat writes
- lower memory wait time in the RTL

What it will not fix by itself:

- software packing overhead
- one-request-at-a-time TL issue behavior
- command overhead outside the datapath

Expected outcome:

- likely another measurable speedup
- not a full `2x` on end-to-end runtime

Main tradeoff:

- wider buses can increase area and routing pressure, and may reduce achievable frequency depending on target and memory path

Recommended way to test it:

- do not overwrite the current `128-bit` config
- use the new `chipyard.MatmulAccel4x4WS256Config` and compare counters side by side against `chipyard.MatmulAccel4x4WSConfig`

## Next Candidate Experiments

1. Keep the prepared-serial weight path as the default reuse API for now.
2. If another RTL pass is still desired, consider limited outstanding C writes or a more aggressive A-side read path on the `256-bit` config.
3. For GPT-style use, structure kernels around prepared weights first; that gives real wins without another hardware redesign.
4. If packing remains a first-order bottleneck even with weight reuse, move toward a raw/strided matrix access path in hardware instead of requiring prepacked streams.

## Current RTL Opportunities

### High-value

1. Wider system bus variant.

- Add a separate `256-bit` WS config instead of replacing the current `128-bit` one.
- Expected benefit: fewer TL beats for A, B, and C traffic.
- Expected limitation: this does not remove software packing cost.

2. Multiple outstanding TL reads.

- The current WS RTL still uses `fromSource = 0` and effectively serializes request/response handling.
- A better version would give the RoCC TL node multiple source IDs and track a small number of in-flight A/B reads.
- This is the cleanest RTL-side latency-hiding improvement still available.

3. Raw/strided matrix access in hardware.

- The current software still packs `4xK` and `Kx4` streams explicitly.
- A more generic accelerator path would accept row-major base pointers plus strides and pack tiles inside hardware.
- This is the most relevant long-term path for later GPT-2 forward-pass work because it removes software marshalling from the hot path.

### Medium-value

4. B-side overlap across `n` tiles.

- Current `wait_fill_b_cycles` is smaller than the A-side wait, but it is still visible.
- Prefetching the next B tile while the current `n0` tile is running could help, especially on larger shapes.

5. Separate persistent weight slots.

- The current preload and `run_preloaded` flow already supports reuse, but the cache is still just one working set.
- If later workloads repeatedly switch among a few weight tiles, explicit weight slots could reduce reload churn.

### Lower-value right now

6. More PE/datapath changes inside the 4x4 array.

- Current counters do not point to the PE mesh as the primary bottleneck.
- The remaining wins are more likely to come from memory/control behavior and software-interface cleanup than from changing the multiply-accumulate datapath itself.
