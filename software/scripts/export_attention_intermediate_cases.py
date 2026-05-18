#!/usr/bin/env python3
"""Export attention debug cases with intermediate score/probability references."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch

SCORE_FRAC_BITS = 16
PROB_FRAC_BITS = 54
BF16_FIXED_INT_BITS = 8
SCORE_SCALE_INT_BITS = 8
ACCUM_BITS = 64


@dataclass(frozen=True)
class AttentionCase:
    name: str
    q_rows: int
    kv_rows: int
    d_k: int
    value_cols: int
    seed: int
    output_tolerance_x100000: int = 25000
    prob_tolerance_x100000: int = 25000


DEFAULT_CASES = (
    AttentionCase("tiny_sanity", 8, 8, 16, 8, 101),
    AttentionCase("gpt2_decode_tile", 8, 256, 64, 64, 102),
    AttentionCase("gpt2_prefill_tile", 128, 128, 64, 64, 103),
    AttentionCase("vit_patch_attention", 197, 197, 64, 64, 104),
)


def bf16_bits(tensor) -> list[int]:
    bf16 = tensor.detach().to(torch.bfloat16).contiguous().cpu()
    return [int(x) & 0xFFFF for x in bf16.view(torch.int16).reshape(-1).tolist()]


def bf16_scalar_bits(value: float) -> int:
    return bf16_bits(torch.tensor([value], dtype=torch.float32))[0]


def bf16_bit_to_float(bits: int) -> float:
    signed_bits = bits if bits < 0x8000 else bits - 0x10000
    return (
        torch.tensor([signed_bits], dtype=torch.int16)
        .view(torch.bfloat16)
        .float()
        .item()
    )


def bf16_to_sint_fixed(bits: int, int_bits: int, frac_bits: int) -> int:
    total_bits = int_bits + frac_bits
    sign = (bits >> 15) & 1
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F

    if (bits & 0x7FFF) == 0:
        return 0

    ext_mantissa = mantissa if exp == 0 else (0x80 | mantissa)
    eff_exp = (1 if exp == 0 else exp) - 127
    shift_amt = eff_exp - 7 + frac_bits
    mag = ext_mantissa << shift_amt if shift_amt >= 0 else ext_mantissa >> (-shift_amt)

    max_pos = (1 << (total_bits - 1)) - 1
    max_neg = 1 << (total_bits - 1)
    if sign == 0 and mag > max_pos:
        mag = max_pos
    elif sign == 1 and mag > max_neg:
        mag = max_neg

    return -mag if sign and mag != 0 else mag


def wrap_sint(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


def fixed_scores(
    q_bits: list[int],
    k_bits: list[int],
    scale_bf16: int,
    case: AttentionCase,
) -> list[int]:
    q_fixed = [
        bf16_to_sint_fixed(x, BF16_FIXED_INT_BITS, 8)
        for x in q_bits
    ]
    k_fixed = [
        bf16_to_sint_fixed(x, BF16_FIXED_INT_BITS, 8)
        for x in k_bits
    ]
    scale_fixed = bf16_to_sint_fixed(scale_bf16, SCORE_SCALE_INT_BITS, SCORE_FRAC_BITS)

    scores: list[int] = []
    for qr in range(case.q_rows):
        for kv in range(case.kv_rows):
            acc = 0
            for d in range(case.d_k):
                acc = wrap_sint(
                    acc + q_fixed[qr * case.d_k + d] * k_fixed[kv * case.d_k + d],
                    ACCUM_BITS,
                )
            scaled = wrap_sint((acc * scale_fixed) >> SCORE_FRAC_BITS, ACCUM_BITS)
            scores.append(scaled)
    return scores


def ideal_probs_fixed(scores: list[int], case: AttentionCase) -> tuple[list[int], torch.Tensor]:
    score_tensor = torch.tensor(scores, dtype=torch.float64).reshape(case.q_rows, case.kv_rows)
    score_tensor = score_tensor / float(1 << SCORE_FRAC_BITS)
    probs = torch.softmax(score_tensor, dim=-1)
    prob_fixed = [
        min((1 << 64) - 1, max(0, int(round(x * float(1 << PROB_FRAC_BITS)))))
        for x in probs.reshape(-1).tolist()
    ]
    return prob_fixed, probs.to(torch.float32)


def emit_u16_array(out, name: str, values: list[int]) -> None:
    out.write(f"const uint16_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 12):
        out.write("  " + ", ".join(f"0x{x & 0xFFFF:04x}" for x in values[i : i + 12]) + ",\n")
    out.write("};\n\n")


def emit_i64_array(out, name: str, values: list[int]) -> None:
    out.write(f"const int64_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 6):
        out.write("  " + ", ".join(f"{x}LL" for x in values[i : i + 6]) + ",\n")
    out.write("};\n\n")


def emit_u64_array(out, name: str, values: list[int]) -> None:
    out.write(f"const uint64_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 4):
        out.write("  " + ", ".join(f"{x}ULL" for x in values[i : i + 4]) + ",\n")
    out.write("};\n\n")


def make_case(case: AttentionCase):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    q = torch.randn(case.q_rows, case.d_k, generator=generator, dtype=torch.float32)
    k = torch.randn(case.kv_rows, case.d_k, generator=generator, dtype=torch.float32)
    v = torch.randn(case.kv_rows, case.value_cols, generator=generator, dtype=torch.float32)

    q_bits = bf16_bits(q)
    k_bits = bf16_bits(k)
    v_bits = bf16_bits(v)
    scale_bf16 = bf16_scalar_bits(1.0 / math.sqrt(float(case.d_k)))

    scores = fixed_scores(q_bits, k_bits, scale_bf16, case)
    probs_fixed, probs_float = ideal_probs_fixed(scores, case)
    v_bf16_float = torch.tensor(
        [bf16_bit_to_float(x) for x in v_bits],
        dtype=torch.float32,
    ).reshape(case.kv_rows, case.value_cols)
    ref = torch.matmul(probs_float, v_bf16_float).to(torch.bfloat16)

    return {
        "q": q_bits,
        "k": k_bits,
        "v": v_bits,
        "ref": bf16_bits(ref),
        "scores": scores,
        "probs": probs_fixed,
        "scale_bf16": scale_bf16,
    }


def print_case_samples(case: AttentionCase, data) -> None:
    mid_score_idx = (case.q_rows // 2) * case.kv_rows + (case.kv_rows // 2)
    print(
        "SAMPLE,"
        f"{case.name},"
        f"scale_bf16=0x{data['scale_bf16']:04x},"
        f"q00={bf16_bit_to_float(data['q'][0]):.6f},"
        f"k00={bf16_bit_to_float(data['k'][0]):.6f},"
        f"v00={bf16_bit_to_float(data['v'][0]):.6f},"
        f"score00={data['scores'][0] / float(1 << SCORE_FRAC_BITS):.6f},"
        f"score_mid={data['scores'][mid_score_idx] / float(1 << SCORE_FRAC_BITS):.6f},"
        f"prob00={data['probs'][0] / float(1 << PROB_FRAC_BITS):.6f},"
        f"out00={bf16_bit_to_float(data['ref'][0]):.6f}"
    )


def write_assets(cases: tuple[AttentionCase, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "attention_intermediate_cases.h"
    source = out_dir / "attention_intermediate_cases.c"

    max_q_rows = max(case.q_rows for case in cases)
    max_kv_rows = max(case.kv_rows for case in cases)
    max_d_k = max(case.d_k for case in cases)
    max_value_cols = max(case.value_cols for case in cases)
    generated = [(case, make_case(case)) for case in cases]

    print("SAMPLE_HEADER,name,scale_bf16,q00,k00,v00,score00,score_mid,prob00,out00")
    for case, data in generated:
        print_case_samples(case, data)

    with header.open("w", encoding="utf-8") as out:
        out.write("#ifndef GIRDAP_GENERATED_ATTENTION_INTERMEDIATE_CASES_H\n")
        out.write("#define GIRDAP_GENERATED_ATTENTION_INTERMEDIATE_CASES_H\n\n")
        out.write("#include <stdint.h>\n\n")
        out.write(f"#define ATTN_INTERMEDIATE_MAX_Q_ROWS {max_q_rows}\n")
        out.write(f"#define ATTN_INTERMEDIATE_MAX_KV_ROWS {max_kv_rows}\n")
        out.write(f"#define ATTN_INTERMEDIATE_MAX_D_K {max_d_k}\n")
        out.write(f"#define ATTN_INTERMEDIATE_MAX_VALUE_COLS {max_value_cols}\n")
        out.write(f"#define ATTN_INTERMEDIATE_SCORE_FRAC_BITS {SCORE_FRAC_BITS}\n")
        out.write(f"#define ATTN_INTERMEDIATE_PROB_FRAC_BITS {PROB_FRAC_BITS}\n\n")
        out.write("typedef struct {\n")
        out.write("  const char *name;\n")
        out.write("  int q_rows;\n")
        out.write("  int kv_rows;\n")
        out.write("  int d_k;\n")
        out.write("  int value_cols;\n")
        out.write("  uint16_t scale_bf16;\n")
        out.write("  uint32_t output_tolerance_x100000;\n")
        out.write("  uint32_t prob_tolerance_x100000;\n")
        out.write("  const uint16_t *q;\n")
        out.write("  const uint16_t *k;\n")
        out.write("  const uint16_t *v;\n")
        out.write("  const uint16_t *expected;\n")
        out.write("  const int64_t *score_fixed;\n")
        out.write("  const uint64_t *prob_fixed;\n")
        out.write("} attention_intermediate_case_t;\n\n")
        out.write("extern const attention_intermediate_case_t attention_intermediate_cases[];\n")
        out.write("extern const int attention_intermediate_case_count;\n\n")
        out.write("#endif\n")

    with source.open("w", encoding="utf-8") as out:
        out.write('#include "attention_intermediate_cases.h"\n\n')
        for idx, (case, data) in enumerate(generated):
            prefix = f"attention_intermediate_case_{idx}"
            out.write(f"/* {case.name}: Q[{case.q_rows},{case.d_k}], ")
            out.write(f"K[{case.kv_rows},{case.d_k}], V[{case.kv_rows},{case.value_cols}] */\n")
            emit_u16_array(out, f"{prefix}_q", data["q"])
            emit_u16_array(out, f"{prefix}_k", data["k"])
            emit_u16_array(out, f"{prefix}_v", data["v"])
            emit_u16_array(out, f"{prefix}_expected", data["ref"])
            emit_i64_array(out, f"{prefix}_score_fixed", data["scores"])
            emit_u64_array(out, f"{prefix}_prob_fixed", data["probs"])

        out.write("const attention_intermediate_case_t attention_intermediate_cases[] = {\n")
        for idx, (case, data) in enumerate(generated):
            prefix = f"attention_intermediate_case_{idx}"
            out.write("  {\n")
            out.write(f'    .name = "{case.name}",\n')
            out.write(f"    .q_rows = {case.q_rows},\n")
            out.write(f"    .kv_rows = {case.kv_rows},\n")
            out.write(f"    .d_k = {case.d_k},\n")
            out.write(f"    .value_cols = {case.value_cols},\n")
            out.write(f"    .scale_bf16 = 0x{data['scale_bf16']:04x},\n")
            out.write(f"    .output_tolerance_x100000 = {case.output_tolerance_x100000},\n")
            out.write(f"    .prob_tolerance_x100000 = {case.prob_tolerance_x100000},\n")
            out.write(f"    .q = {prefix}_q,\n")
            out.write(f"    .k = {prefix}_k,\n")
            out.write(f"    .v = {prefix}_v,\n")
            out.write(f"    .expected = {prefix}_expected,\n")
            out.write(f"    .score_fixed = {prefix}_score_fixed,\n")
            out.write(f"    .prob_fixed = {prefix}_prob_fixed,\n")
            out.write("  },\n")
        out.write("};\n\n")
        out.write(
            "const int attention_intermediate_case_count = "
            "sizeof(attention_intermediate_cases) / sizeof(attention_intermediate_cases[0]);\n"
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
    print(f"wrote attention intermediate cases to {args.out_dir} using PyTorch backend")


if __name__ == "__main__":
    main()
