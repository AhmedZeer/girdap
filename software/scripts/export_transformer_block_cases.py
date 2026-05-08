#!/usr/bin/env python3
"""Export tiny transformer attention-block cases as linkable C assets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TransformerBlockCase:
    name: str
    seq_len: int
    d_model: int
    n_heads: int
    seed: int
    input_scale: float = 0.5
    weight_scale: float = 0.0625
    tolerance_x100000: int = 50000

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


DEFAULT_CASES = (
    TransformerBlockCase("tiny_seq8_dm32_h4", 8, 32, 4, 301),
    TransformerBlockCase("tiny_seq16_dm64_h4", 16, 64, 4, 302),
    TransformerBlockCase("rect_seq32_dm64_h4", 32, 64, 4, 303),
    TransformerBlockCase("seq32_dm128_h4", 32, 128, 4, 304),
    TransformerBlockCase("seq64_dm128_h4", 64, 128, 4, 305),
    TransformerBlockCase("seq64_dm256_h4", 64, 256, 4, 306),
    TransformerBlockCase("seq128_dm256_h4", 128, 256, 4, 307),
)


def bf16_bits(tensor: torch.Tensor) -> list[int]:
    bf16 = tensor.detach().to(torch.bfloat16).contiguous().cpu()
    return [int(x) & 0xFFFF for x in bf16.view(torch.int16).reshape(-1).tolist()]


def bf16_scalar_bits(value: float) -> int:
    return bf16_bits(torch.tensor([value], dtype=torch.float32))[0]


def bf16_scalar_float(value: float) -> float:
    return torch.tensor([value], dtype=torch.float32).to(torch.bfloat16).float().item()


def bf16_bit_to_float(bits: int) -> float:
    signed_bits = bits if bits < 0x8000 else bits - 0x10000
    return (
        torch.tensor([signed_bits], dtype=torch.int16)
        .view(torch.bfloat16)
        .float()
        .item()
    )


def emit_array(out, name: str, values: list[int]) -> None:
    out.write(f"const uint16_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 12):
        chunk = values[i : i + 12]
        out.write("  " + ", ".join(f"0x{x:04x}" for x in chunk) + ",\n")
    out.write("};\n\n")


def uniform_scaled(
    generator: torch.Generator,
    rows: int,
    cols: int,
    scale: float,
) -> torch.Tensor:
    return (torch.rand((rows, cols), generator=generator, dtype=torch.float32) * 2.0 - 1.0) * scale


def make_case(case: TransformerBlockCase) -> dict[str, list[int] | int]:
    if case.d_model % case.n_heads != 0:
        raise ValueError(f"{case.name}: d_model must be divisible by n_heads")
    if case.head_dim > 256:
        raise ValueError(f"{case.name}: head_dim exceeds hardware maxK")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    x = uniform_scaled(generator, case.seq_len, case.d_model, case.input_scale)
    wq = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale)
    wk = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale)
    wv = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale)
    wo = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale)

    x_bf16 = x.to(torch.bfloat16)
    wq_bf16 = wq.to(torch.bfloat16)
    wk_bf16 = wk.to(torch.bfloat16)
    wv_bf16 = wv.to(torch.bfloat16)
    wo_bf16 = wo.to(torch.bfloat16)

    # Match the hardware-visible datatype boundaries: each projection emits BF16,
    # attention consumes BF16 per head and emits BF16, then output projection emits BF16.
    q = torch.matmul(x_bf16.float(), wq_bf16.float()).to(torch.bfloat16)
    k = torch.matmul(x_bf16.float(), wk_bf16.float()).to(torch.bfloat16)
    v = torch.matmul(x_bf16.float(), wv_bf16.float()).to(torch.bfloat16)

    scale_value = 1.0 / math.sqrt(float(case.head_dim))
    scale_bf16 = bf16_scalar_bits(scale_value)
    scale_float = bf16_scalar_float(scale_value)

    context_heads = []
    for h in range(case.n_heads):
        start = h * case.head_dim
        end = start + case.head_dim
        qh = q[:, start:end].float()
        kh = k[:, start:end].float()
        vh = v[:, start:end].float()
        scores = torch.matmul(qh, kh.transpose(0, 1)) * scale_float
        probs = torch.softmax(scores, dim=-1)
        context_heads.append(torch.matmul(probs, vh).to(torch.bfloat16))

    context = torch.cat(context_heads, dim=1).to(torch.bfloat16)
    output = torch.matmul(context.float(), wo_bf16.float()).to(torch.bfloat16)

    return {
        "x": bf16_bits(x_bf16),
        "wq": bf16_bits(wq_bf16),
        "wk": bf16_bits(wk_bf16),
        "wv": bf16_bits(wv_bf16),
        "wo": bf16_bits(wo_bf16),
        "expected": bf16_bits(output),
        "scale_bf16": scale_bf16,
    }


def print_case_sample(case: TransformerBlockCase, data: dict[str, list[int] | int]) -> None:
    exp = data["expected"]
    mid_idx = (case.seq_len // 2) * case.d_model + (case.d_model // 2)
    print(
        "SAMPLE,"
        f"{case.name},"
        f"seq={case.seq_len},"
        f"d_model={case.d_model},"
        f"heads={case.n_heads},"
        f"head_dim={case.head_dim},"
        f"scale_bf16=0x{data['scale_bf16']:04x},"
        f"x00={bf16_bit_to_float(data['x'][0]):.6f},"
        f"wq00={bf16_bit_to_float(data['wq'][0]):.6f},"
        f"ref00={bf16_bit_to_float(exp[0]):.6f},"
        f"ref_mid={bf16_bit_to_float(exp[mid_idx]):.6f},"
        f"ref_last={bf16_bit_to_float(exp[-1]):.6f}"
    )


def write_assets(cases: tuple[TransformerBlockCase, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "transformer_block_cases.h"
    source = out_dir / "transformer_block_cases.c"

    max_seq_len = max(case.seq_len for case in cases)
    max_d_model = max(case.d_model for case in cases)
    max_heads = max(case.n_heads for case in cases)
    max_head_dim = max(case.head_dim for case in cases)

    generated = [(case, make_case(case)) for case in cases]

    print("SAMPLE_HEADER,name,seq,d_model,heads,head_dim,scale_bf16,x00,wq00,ref00,ref_mid,ref_last")
    for case, data in generated:
        print_case_sample(case, data)

    with header.open("w", encoding="utf-8") as out:
        out.write("#ifndef TOYROCC_GENERATED_TRANSFORMER_BLOCK_CASES_H\n")
        out.write("#define TOYROCC_GENERATED_TRANSFORMER_BLOCK_CASES_H\n\n")
        out.write("#include <stdint.h>\n\n")
        out.write(f"#define TRANSFORMER_BLOCK_MAX_SEQ_LEN {max_seq_len}\n")
        out.write(f"#define TRANSFORMER_BLOCK_MAX_D_MODEL {max_d_model}\n")
        out.write(f"#define TRANSFORMER_BLOCK_MAX_HEADS {max_heads}\n")
        out.write(f"#define TRANSFORMER_BLOCK_MAX_HEAD_DIM {max_head_dim}\n\n")
        out.write("typedef struct {\n")
        out.write("  const char *name;\n")
        out.write("  int seq_len;\n")
        out.write("  int d_model;\n")
        out.write("  int n_heads;\n")
        out.write("  int head_dim;\n")
        out.write("  uint16_t scale_bf16;\n")
        out.write("  uint32_t tolerance_x100000;\n")
        out.write("  const uint16_t *x;\n")
        out.write("  const uint16_t *wq;\n")
        out.write("  const uint16_t *wk;\n")
        out.write("  const uint16_t *wv;\n")
        out.write("  const uint16_t *wo;\n")
        out.write("  const uint16_t *expected;\n")
        out.write("} transformer_block_case_t;\n\n")
        out.write("extern const transformer_block_case_t transformer_block_cases[];\n")
        out.write("extern const int transformer_block_case_count;\n\n")
        out.write("#endif\n")

    with source.open("w", encoding="utf-8") as out:
        out.write('#include "transformer_block_cases.h"\n\n')
        for idx, (case, data) in enumerate(generated):
            prefix = f"transformer_block_case_{idx}"
            out.write(
                f"/* {case.name}: X[{case.seq_len},{case.d_model}], "
                f"heads={case.n_heads}, head_dim={case.head_dim} */\n"
            )
            emit_array(out, f"{prefix}_x", data["x"])
            emit_array(out, f"{prefix}_wq", data["wq"])
            emit_array(out, f"{prefix}_wk", data["wk"])
            emit_array(out, f"{prefix}_wv", data["wv"])
            emit_array(out, f"{prefix}_wo", data["wo"])
            emit_array(out, f"{prefix}_expected", data["expected"])

        out.write("const transformer_block_case_t transformer_block_cases[] = {\n")
        for idx, (case, data) in enumerate(generated):
            prefix = f"transformer_block_case_{idx}"
            out.write("  {\n")
            out.write(f'    .name = "{case.name}",\n')
            out.write(f"    .seq_len = {case.seq_len},\n")
            out.write(f"    .d_model = {case.d_model},\n")
            out.write(f"    .n_heads = {case.n_heads},\n")
            out.write(f"    .head_dim = {case.head_dim},\n")
            out.write(f"    .scale_bf16 = 0x{data['scale_bf16']:04x},\n")
            out.write(f"    .tolerance_x100000 = {case.tolerance_x100000},\n")
            out.write(f"    .x = {prefix}_x,\n")
            out.write(f"    .wq = {prefix}_wq,\n")
            out.write(f"    .wk = {prefix}_wk,\n")
            out.write(f"    .wv = {prefix}_wv,\n")
            out.write(f"    .wo = {prefix}_wo,\n")
            out.write(f"    .expected = {prefix}_expected,\n")
            out.write("  },\n")
        out.write("};\n\n")
        out.write(
            "const int transformer_block_case_count = "
            "sizeof(transformer_block_cases) / sizeof(transformer_block_cases[0]);\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "generated",
        help="Directory for generated C assets.",
    )
    args = parser.parse_args()
    write_assets(DEFAULT_CASES, args.out_dir)
    print(f"wrote transformer block cases to {args.out_dir} using PyTorch backend")


if __name__ == "__main__":
    main()
