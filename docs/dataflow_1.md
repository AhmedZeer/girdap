# Weight-Stationary Dataflow 1

This note explains the visual meaning of:

- tile
- chunk
- `aChunkBufs`
- `bCacheBuf`
- `weightTile`
- `currentInVec`
- the main indices used by the WS wrapper

The examples below assume the current `4 x 4` weight-stationary array.

## 1. Full GEMM

We compute:

```text
C[M x N] = A[M x K] x B[K x N]
```

Example:

- `M = 8`
- `N = 8`
- `K = 10`
- array = `4 x 4`

So:

- `A` is `8 x 10`
- `B` is `10 x 8`
- `C` is `8 x 8`

## 2. What Is A Tile?

Because the array is `4 x 4`, software works on `4 x 4` output tiles.

That means:

- one A tile is `4 x K`
- one B tile is `K x 4`
- one output tile is `4 x 4`

For the `8 x 8 x 10` example:

```text
A = 8 x 10  -> split into two row tiles
B = 10 x 8  -> split into two column tiles
C = 8 x 8   -> split into four 4x4 output tiles
```

Visual:

```text
A (8 x 10)

rows 0..3   -> A tile 0 : 4 x 10
rows 4..7   -> A tile 1 : 4 x 10
```

```text
B (10 x 8)

cols 0..3   -> B tile 0 : 10 x 4
cols 4..7   -> B tile 1 : 10 x 4
```

Then outputs are:

```text
C tile (0,0) = A tile 0 x B tile 0   -> 4 x 4
C tile (0,1) = A tile 0 x B tile 1   -> 4 x 4
C tile (1,0) = A tile 1 x B tile 0   -> 4 x 4
C tile (1,1) = A tile 1 x B tile 1   -> 4 x 4
```

So:

- tile = matrix-level blocking for the `4 x 4` array

## 3. What Is A Chunk?

A tile can still have large `K`.

The hardware cannot consume all `K` at once.

In this design:

```text
chunkKMax = nRows = 4
```

So the tile's K dimension is broken into chunks of at most 4.

For one A tile `4 x 10` and one B tile `10 x 4`:

```text
K = 10
chunkKMax = 4
```

So the tile is processed in 3 chunks:

- chunk 0: K indices `0..3`
- chunk 1: K indices `4..7`
- chunk 2: K indices `8..9`

Visual:

```text
A tile (4 x 10)

      k0 k1 k2 k3 | k4 k5 k6 k7 | k8 k9
r0     .  .  .  . |  .  .  .  . |  .  .
r1     .  .  .  . |  .  .  .  . |  .  .
r2     .  .  .  . |  .  .  .  . |  .  .
r3     .  .  .  . |  .  .  .  . |  .  .
        chunk 0      chunk 1      chunk 2
```

```text
B tile (10 x 4)

      c0 c1 c2 c3
k0     .  .  .  .
k1     .  .  .  .
k2     .  .  .  .
k3     .  .  .  .
-----------------
k4     .  .  .  .
k5     .  .  .  .
k6     .  .  .  .
k7     .  .  .  .
-----------------
k8     .  .  .  .
k9     .  .  .  .
```

So:

- chunk = K-slice of one A/B tile pair

## 4. What Is `bCacheBuf`?

`bCacheBuf` stores the packed B stream for the whole current B tile.

For our `10 x 4` B tile:

- one packed word per K-step
- each packed word contains 4 column values

So:

```text
bCacheBuf[0] = [B[0][0], B[0][1], B[0][2], B[0][3]]
bCacheBuf[1] = [B[1][0], B[1][1], B[1][2], B[1][3]]
bCacheBuf[2] = [B[2][0], B[2][1], B[2][2], B[2][3]]
...
bCacheBuf[9] = [B[9][0], B[9][1], B[9][2], B[9][3]]
```

Visual:

```text
bCacheBuf

idx  packed contents
0    [k0,c0  k0,c1  k0,c2  k0,c3]
1    [k1,c0  k1,c1  k1,c2  k1,c3]
2    [k2,c0  k2,c1  k2,c2  k2,c3]
3    [k3,c0  k3,c1  k3,c2  k3,c3]
4    [k4,c0  k4,c1  k4,c2  k4,c3]
...
9    [k9,c0  k9,c1  k9,c2  k9,c3]
```

So:

- B cache = local packed storage for one whole `K x 4` B tile

It is called "cache" informally, but it is really a software-managed local buffer.

## 5. What Is `aChunkBufs`?

`aChunkBufs` is the ping-pong storage for one A chunk at a time, not the whole A tile.

Since `chunkKMax = 4`, one chunk needs at most 4 packed A words.

Each packed A word contains one K-step across all 4 rows:

```text
aChunk word 0 = [A[0][k], A[1][k], A[2][k], A[3][k]]
```

For chunk 0 (`k=0..3`):

```text
buffer slot 0 = [A[0][0], A[1][0], A[2][0], A[3][0]]
buffer slot 1 = [A[0][1], A[1][1], A[2][1], A[3][1]]
buffer slot 2 = [A[0][2], A[1][2], A[2][2], A[3][2]]
buffer slot 3 = [A[0][3], A[1][3], A[2][3], A[3][3]]
```

And there are two such buffers:

- one active for compute
- one for prefetch

Visual:

```text
aChunkBufs(0)   active or prefetched
0  [r0k0 r1k0 r2k0 r3k0]
1  [r0k1 r1k1 r2k1 r3k1]
2  [r0k2 r1k2 r2k2 r3k2]
3  [r0k3 r1k3 r2k3 r3k3]
```

```text
aChunkBufs(1)   the other ping-pong half
0  [...]
1  [...]
2  [...]
3  [...]
```

So:

- A chunk buffer = local packed storage for one K-chunk of one A tile

## 6. What Is `weightTile`?

`weightTile` is the actual `4 x 4` set of weights loaded into the mesh for the current chunk.

It is extracted from `bCacheBuf` using `kBaseIdx`.

For chunk 0:

```text
weightTile =
rows from bCacheBuf[0..3]
```

Visual:

```text
weightTile for chunk 0

         c0 c1 c2 c3
k0/r0     .  .  .  .
k1/r1     .  .  .  .
k2/r2     .  .  .  .
k3/r3     .  .  .  .
```

For chunk 1:

```text
weightTile =
rows from bCacheBuf[4..7]
```

So:

- `weightTile` = one `chunkK x 4` slice of the B tile, zero-padded to `4 x 4` if needed

## 7. What Is `currentInVec`?

`currentInVec` is the per-cycle vector fed into the array.

For one cycle, it contains 4 A values:

```text
currentInVec = [a0, a1, a2, a3]
```

These are taken from one row of the unpacked chunk view.

If chunk 0 is loaded:

```text
aChunkBufs active

0  [A0k0 A1k0 A2k0 A3k0]
1  [A0k1 A1k1 A2k1 A3k1]
2  [A0k2 A1k2 A2k2 A3k2]
3  [A0k3 A1k3 A2k3 A3k3]
```

Then during feed:

- cycle 0:
  - `currentInVec = [A0k0 A1k0 A2k0 A3k0]`
- cycle 1:
  - `currentInVec = [A0k1 A1k1 A2k1 A3k1]`
- cycle 2:
  - `currentInVec = [A0k2 A1k2 A2k2 A3k2]`
- cycle 3:
  - `currentInVec = [A0k3 A1k3 A2k3 A3k3]`

So:

- `currentInVec` = one streamed A vector for one cycle

## 8. What Do The Important Indices Mean?

Let's stay with:

- one A tile = `4 x 10`
- one B tile = `10 x 4`
- chunks = `[0..3]`, `[4..7]`, `[8..9]`

`kBaseIdx`

- where current chunk starts inside full K
- chunk 0 -> `0`
- chunk 1 -> `4`
- chunk 2 -> `8`

`chunkK`

- size of current chunk
- chunk 0 -> `4`
- chunk 1 -> `4`
- chunk 2 -> `2`

`fillIdx`

- which entry of current A chunk buffer is being filled
- during loading of chunk 0:
  - `0,1,2,3`

`prefetchFillIdx`

- same idea, but for the other ping-pong buffer during prefetch

`feedRowIdx`

- which packed A word from the active chunk is being fed this cycle
- during chunk 0 compute:
  - `0,1,2,3`

`captureRowIdx`

- which output row from the chunk is currently being accumulated into `accumTile`
- because outputs return over time after the systolic latency

`outIdx`

- which output word is being written back to memory
- not matrix row/col directly, but linear writeback word index

`bCacheFillIdx`

- how much of `bCacheBuf` has been committed so far during B loading

`bIssueIdx`

- next B-cache word index to issue as a memory read request

## 9. Full Picture In One Diagram

For one output tile:

```text
A tile: 4 x 10
B tile: 10 x 4
C tile: 4 x 4
```

```text
Step 1: load full B tile into bCacheBuf

bCacheBuf:
0  [B0,0 B0,1 B0,2 B0,3]
1  [B1,0 B1,1 B1,2 B1,3]
2  [B2,0 B2,1 B2,2 B2,3]
3  [B3,0 B3,1 B3,2 B3,3]
4  [B4,0 B4,1 B4,2 B4,3]
...
9  [B9,0 B9,1 B9,2 B9,3]
```

```text
Step 2: chunk 0 uses K=0..3

weightTile:
from bCacheBuf[0..3]

aChunkBuf active:
0  [A0,0 A1,0 A2,0 A3,0]
1  [A0,1 A1,1 A2,1 A3,1]
2  [A0,2 A1,2 A2,2 A3,2]
3  [A0,3 A1,3 A2,3 A3,3]
```

```text
Step 3: feed chunk 0 over time

cycle 0 -> currentInVec = [A0,0 A1,0 A2,0 A3,0]
cycle 1 -> currentInVec = [A0,1 A1,1 A2,1 A3,1]
cycle 2 -> currentInVec = [A0,2 A1,2 A2,2 A3,2]
cycle 3 -> currentInVec = [A0,3 A1,3 A2,3 A3,3]
```

```text
Step 4: while chunk 0 computes, prefetch chunk 1 into other aChunkBuf

other aChunkBuf:
0  [A0,4 A1,4 A2,4 A3,4]
1  [A0,5 A1,5 A2,5 A3,5]
2  [A0,6 A1,6 A2,6 A3,6]
3  [A0,7 A1,7 A2,7 A3,7]
```

```text
Step 5: accumulate outputs of all chunks into accumTile

accumTile:
4 x 4 partial sums
```

```text
Step 6: after all chunks done, write accumTile -> memory
```

## 10. Shortest Summary

- tile = matrix block matched to array shape
  - A tile = `4 x K`
  - B tile = `K x 4`
  - C tile = `4 x 4`

- chunk = smaller K-slice of one tile pair
  - because hardware only handles up to `chunkKMax = 4` K-steps at once

- `bCacheBuf` = full packed B tile stored locally
- `aChunkBufs` = ping-pong packed A chunk buffers
- `weightTile` = current `4 x 4` weight slice loaded into mesh
- `currentInVec` = per-cycle 4-lane A vector fed into mesh

## Notes
- B buffer is for reuse (load whole TILE)
- A ping-pong buffers are for streaming and overlap (load whole CHUNK)
