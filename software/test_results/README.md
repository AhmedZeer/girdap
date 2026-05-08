# Dual-RoCC Performance Summary

Speedups are end-to-end workload speedups from the `speedup_x100` column in the
`dual_accel*` result files, reported as `cpu_cycles / hw_e2e_cycles`.

| Model | Shapes | Speedup |
|---|---|---:|
| Tiny-BERT inference | L1, seq=32, d_model=128, heads=4, head_dim=32, FFN=256, vocab=512, classes=2 | 13.91x |
| Tiny-BERT inference | L2, seq=64, d_model=128, heads=4, head_dim=32, FFN=256, vocab=512, classes=2 | 18.49x |
| Tiny-BERT inference | L4, seq=128, d_model=128, heads=4, head_dim=32, FFN=256, vocab=512, classes=2 | 23.74x |
| ViT-style encoder | L1, seq=65, d_model=128, heads=4, head_dim=32, FFN=256 | 19.90x |
| ViT-style encoder | L2, seq=65, d_model=128, heads=4, head_dim=32, FFN=256 | 19.93x |
| ViT-style encoder | L2, seq=197, d_model=256, heads=4, head_dim=64, FFN=256 | 49.95x |
| Transformer attention block | seq=8, d_model=32, heads=4, head_dim=8 | 5.81x |
| Transformer attention block | seq=16, d_model=64, heads=4, head_dim=16 | 10.23x |
| Transformer attention block | seq=32, d_model=64, heads=4, head_dim=16 | 15.74x |
| Transformer attention block | seq=32, d_model=128, heads=4, head_dim=32 | 18.04x |
| Transformer attention block | seq=64, d_model=128, heads=4, head_dim=32 | 27.18x |
| Transformer attention block | seq=64, d_model=256, heads=4, head_dim=64 | 47.14x |
| Transformer attention block | seq=128, d_model=256, heads=4, head_dim=64 | 63.53x |
| Full transformer layer | seq=8, d_model=32, heads=4, head_dim=8, FFN=64 | 4.13x |
| Full transformer layer | seq=16, d_model=64, heads=4, head_dim=16, FFN=128 | 7.30x |
| Full transformer layer | seq=32, d_model=64, heads=4, head_dim=16, FFN=128 | 9.68x |
| Full transformer layer | seq=32, d_model=128, heads=4, head_dim=32, FFN=256 | 15.88x |
| Full transformer layer | seq=64, d_model=128, heads=4, head_dim=32, FFN=256 | 20.30x |
| Full transformer layer | seq=128, d_model=256, heads=4, head_dim=64, FFN=256 | 44.16x |

Notes:

- Tiny-BERT includes token/position/type embeddings, embedding LayerNorm, BERT-style post-LN encoder layers, pooler, and classifier logits.
- ViT-style encoder and transformer rows use synthetic BF16 weights and PyTorch-generated references.
- These results use the current dual-RoCC configuration: matmul on custom0 and attention on custom1.
