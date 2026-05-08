#!/usr/bin/env python3
"""Export PyTorch attention operator cases as linkable C assets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class AttentionCase:
    name: str
    q_rows: int
    kv_rows: int
    d_k: int
    value_cols: int
    seed: int
    input_scale: float = 1.0
    value_scale: float = 1.0
    tolerance_x100000: int = 25000


DEFAULT_CASES = (
    AttentionCase("torch_8x8x16x8_s025", 8, 8, 16, 8, 201, 0.25, 0.25),
    AttentionCase("torch_8x9x16x8_s025", 8, 9, 16, 8, 202, 0.25, 0.25),
    AttentionCase("torch_8x16x16x8_s025", 8, 16, 16, 8, 203, 0.25, 0.25),
    AttentionCase("torch_16x16x16x16_s025", 16, 16, 16, 16, 204, 0.25, 0.25),
    AttentionCase("torch_8x64x64x8_s025", 8, 64, 64, 8, 205, 0.25, 0.25),
    AttentionCase("torch_8x8x16x8_s05", 8, 8, 16, 8, 211, 0.5, 0.5),
    AttentionCase("torch_8x9x16x8_s05", 8, 9, 16, 8, 212, 0.5, 0.5),
    AttentionCase("torch_8x16x16x8_s05", 8, 16, 16, 8, 213, 0.5, 0.5),
    AttentionCase("torch_16x16x16x16_s05", 16, 16, 16, 16, 214, 0.5, 0.5),
    AttentionCase("torch_8x64x64x8_s05", 8, 64, 64, 8, 215, 0.5, 0.5),
    AttentionCase("tiny_sanity", 8, 8, 16, 8, 101),
    AttentionCase("tiny_sanity_s025", 8, 8, 16, 8, 101, 0.25, 0.25),
    AttentionCase("tiny_sanity_s05", 8, 8, 16, 8, 101, 0.5, 0.5),
    AttentionCase("gpt2_decode_tile", 8, 256, 64, 64, 102),
    AttentionCase("gpt2_decode_tile_s025", 8, 256, 64, 64, 102, 0.25, 0.25),
    AttentionCase("gpt2_decode_tile_s05", 8, 256, 64, 64, 102, 0.5, 0.5),
    AttentionCase("gpt2_prefill_tile", 128, 128, 64, 64, 103),
    AttentionCase("gpt2_prefill_tile_s025", 128, 128, 64, 64, 103, 0.25, 0.25),
    AttentionCase("gpt2_prefill_tile_s05", 128, 128, 64, 64, 103, 0.5, 0.5),
    AttentionCase("vit_patch_attention", 197, 197, 64, 64, 104),
    AttentionCase("vit_patch_attention_s025", 197, 197, 64, 64, 104, 0.25, 0.25),
    AttentionCase("vit_patch_attention_s05", 197, 197, 64, 64, 104, 0.5, 0.5),
    AttentionCase("max_shape_stress", 256, 256, 256, 128, 105),
    AttentionCase("max_shape_stress_s025", 256, 256, 256, 128, 105, 0.25, 0.25),
    AttentionCase("max_shape_stress_s05", 256, 256, 256, 128, 105, 0.5, 0.5),
)


def bf16_bits(tensor) -> list[int]:
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


def bf16_bits_to_float_list(bits: list[int], count: int = 5) -> str:
    values = [bf16_bit_to_float(x) for x in bits[:count]]
    return "[" + ",".join(f"{x:.6f}" for x in values) + "]"


def emit_array(out, name: str, values: list[int]) -> None:
    out.write(f"const uint16_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 12):
        chunk = values[i : i + 12]
        out.write("  " + ", ".join(f"0x{x:04x}" for x in chunk) + ",\n")
    out.write("};\n\n")


def print_case_samples(case: AttentionCase, data: dict[str, list[int] | int]) -> None:
    q_head = bf16_bits_to_float_list(data["q"])
    k_head = bf16_bits_to_float_list(data["k"])
    v_head = bf16_bits_to_float_list(data["v"])
    ref_head = bf16_bits_to_float_list(data["ref"])
    ref_mid_idx = (case.q_rows // 2) * case.value_cols + (case.value_cols // 2)
    ref_mid = bf16_bit_to_float(data["ref"][ref_mid_idx])
    ref_last = bf16_bit_to_float(data["ref"][-1])

    print(
        "SAMPLE,\n"
        f"{case.name},\n"
        f"scale_bf16=0x{data['scale_bf16']:04x},\n"
        f"q_head={q_head},\n"
        f"k_head={k_head},\n"
        f"v_head={v_head},\n"
        f"ref_head={ref_head},\n"
        f"ref[mid,mid]={ref_mid:.6f},\n"
        f"ref[last,last]={ref_last:.6f}\n"
        f"\n\n"
    )


def uniform_minus_one_to_one(
    generator: torch.Generator,
    rows: int,
    cols: int,
) -> torch.Tensor:
    return torch.rand((rows, cols), generator=generator, dtype=torch.float32) * 2.0 - 1.0


def make_case(case: AttentionCase):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    q = uniform_minus_one_to_one(generator, case.q_rows, case.d_k) * case.input_scale
    k = uniform_minus_one_to_one(generator, case.kv_rows, case.d_k) * case.input_scale
    v = uniform_minus_one_to_one(generator, case.kv_rows, case.value_cols) * case.value_scale

    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)

    scale_value = 1.0 / math.sqrt(float(case.d_k))
    scale_bf16 = bf16_scalar_bits(scale_value)
    scale_float = bf16_scalar_float(scale_value)

    scores = torch.matmul(q_bf16.float(), k_bf16.float().transpose(0, 1)) * scale_float
    probs = torch.softmax(scores, dim=-1)
    ref = torch.matmul(probs, v_bf16.float()).to(torch.bfloat16)

    return {
        "q": bf16_bits(q_bf16),
        "k": bf16_bits(k_bf16),
        "v": bf16_bits(v_bf16),
        "ref": bf16_bits(ref),
        "scale_bf16": scale_bf16,
    }


def write_assets(cases: tuple[AttentionCase, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "attention_operator_cases.h"
    source = out_dir / "attention_operator_cases.c"

    max_q_rows = max(case.q_rows for case in cases)
    max_kv_rows = max(case.kv_rows for case in cases)
    max_d_k = max(case.d_k for case in cases)
    max_value_cols = max(case.value_cols for case in cases)

    generated = [(case, make_case(case)) for case in cases]

    print("SAMPLE_HEADER,name,scale_bf16,q00,k00,v00,ref00,ref_mid,ref_last")
    for case, data in generated:
        print_case_samples(case, data)

    with header.open("w", encoding="utf-8") as out:
        out.write("#ifndef TOYROCC_GENERATED_ATTENTION_OPERATOR_CASES_H\n")
        out.write("#define TOYROCC_GENERATED_ATTENTION_OPERATOR_CASES_H\n\n")
        out.write("#include <stdint.h>\n\n")
        out.write(f"#define ATTN_OPERATOR_MAX_Q_ROWS {max_q_rows}\n")
        out.write(f"#define ATTN_OPERATOR_MAX_KV_ROWS {max_kv_rows}\n")
        out.write(f"#define ATTN_OPERATOR_MAX_D_K {max_d_k}\n")
        out.write(f"#define ATTN_OPERATOR_MAX_VALUE_COLS {max_value_cols}\n\n")
        out.write("typedef struct {\n")
        out.write("  const char *name;\n")
        out.write("  int q_rows;\n")
        out.write("  int kv_rows;\n")
        out.write("  int d_k;\n")
        out.write("  int value_cols;\n")
        out.write("  uint16_t scale_bf16;\n")
        out.write("  uint32_t tolerance_x100000;\n")
        out.write("  const uint16_t *q;\n")
        out.write("  const uint16_t *k;\n")
        out.write("  const uint16_t *v;\n")
        out.write("  const uint16_t *expected;\n")
        out.write("} attention_operator_case_t;\n\n")
        out.write("extern const attention_operator_case_t attention_operator_cases[];\n")
        out.write("extern const int attention_operator_case_count;\n\n")
        out.write("#endif\n")

    with source.open("w", encoding="utf-8") as out:
        out.write('#include "attention_operator_cases.h"\n\n')
        for idx, (case, data) in enumerate(generated):
            prefix = f"attention_case_{idx}"
            out.write(f"/* {case.name}: Q[{case.q_rows},{case.d_k}], ")
            out.write(f"K[{case.kv_rows},{case.d_k}], V[{case.kv_rows},{case.value_cols}] */\n")
            emit_array(out, f"{prefix}_q", data["q"])
            emit_array(out, f"{prefix}_k", data["k"])
            emit_array(out, f"{prefix}_v", data["v"])
            emit_array(out, f"{prefix}_expected", data["ref"])

        out.write("const attention_operator_case_t attention_operator_cases[] = {\n")
        for idx, (case, data) in enumerate(generated):
            prefix = f"attention_case_{idx}"
            out.write("  {\n")
            out.write(f'    .name = "{case.name}",\n')
            out.write(f"    .q_rows = {case.q_rows},\n")
            out.write(f"    .kv_rows = {case.kv_rows},\n")
            out.write(f"    .d_k = {case.d_k},\n")
            out.write(f"    .value_cols = {case.value_cols},\n")
            out.write(f"    .scale_bf16 = 0x{data['scale_bf16']:04x},\n")
            out.write(f"    .tolerance_x100000 = {case.tolerance_x100000},\n")
            out.write(f"    .q = {prefix}_q,\n")
            out.write(f"    .k = {prefix}_k,\n")
            out.write(f"    .v = {prefix}_v,\n")
            out.write(f"    .expected = {prefix}_expected,\n")
            out.write("  },\n")
        out.write("};\n\n")
        out.write(
            "const int attention_operator_case_count = "
            "sizeof(attention_operator_cases) / sizeof(attention_operator_cases[0]);\n"
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
    print(f"wrote attention operator cases to {args.out_dir} using PyTorch backend")


if __name__ == "__main__":
    main()
