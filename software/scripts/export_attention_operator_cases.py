#!/usr/bin/env python3
"""Export PyTorch attention operator cases as linkable C assets."""

from __future__ import annotations

import argparse
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass(frozen=True)
class AttentionCase:
    name: str
    q_rows: int
    kv_rows: int
    d_k: int
    value_cols: int
    seed: int
    tolerance_x100000: int = 25000


DEFAULT_CASES = (
    AttentionCase("tiny_sanity", 8, 8, 16, 8, 101),
    AttentionCase("gpt2_decode_tile", 8, 256, 64, 64, 102),
    AttentionCase("gpt2_prefill_tile", 128, 128, 64, 64, 103),
    AttentionCase("vit_patch_attention", 197, 197, 64, 64, 104),
    AttentionCase("max_shape_stress", 256, 256, 256, 128, 105),
)


def float_to_bf16_bits(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def bf16_bits_to_float(bits: int) -> float:
    value = (bits & 0xFFFF) << 16
    return struct.unpack(">f", struct.pack(">I", value))[0]


def bf16_bits(tensor) -> list[int]:
    bf16 = tensor.detach().to(torch.bfloat16).contiguous().cpu()
    return [int(x) & 0xFFFF for x in bf16.view(torch.int16).reshape(-1).tolist()]


def bf16_scalar_bits(value: float) -> int:
    if torch is None:
        return float_to_bf16_bits(value)
    return bf16_bits(torch.tensor([value], dtype=torch.float32))[0]


def bf16_scalar_float(value: float) -> float:
    if torch is None:
        return bf16_bits_to_float(float_to_bf16_bits(value))
    return torch.tensor([value], dtype=torch.float32).to(torch.bfloat16).float().item()


def emit_array(out, name: str, values: list[int]) -> None:
    out.write(f"const uint16_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 12):
        chunk = values[i : i + 12]
        out.write("  " + ", ".join(f"0x{x:04x}" for x in chunk) + ",\n")
    out.write("};\n\n")


def make_case_torch(case: AttentionCase):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    q = torch.randn(case.q_rows, case.d_k, generator=generator, dtype=torch.float32)
    k = torch.randn(case.kv_rows, case.d_k, generator=generator, dtype=torch.float32)
    v = torch.randn(case.kv_rows, case.value_cols, generator=generator, dtype=torch.float32)

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


def make_case_stdlib(case: AttentionCase):
    rng = random.Random(case.seed)
    q_bits = [float_to_bf16_bits(rng.gauss(0.0, 1.0)) for _ in range(case.q_rows * case.d_k)]
    k_bits = [float_to_bf16_bits(rng.gauss(0.0, 1.0)) for _ in range(case.kv_rows * case.d_k)]
    v_bits = [
        float_to_bf16_bits(rng.gauss(0.0, 1.0))
        for _ in range(case.kv_rows * case.value_cols)
    ]
    q = [bf16_bits_to_float(x) for x in q_bits]
    k = [bf16_bits_to_float(x) for x in k_bits]
    v = [bf16_bits_to_float(x) for x in v_bits]

    scale_value = 1.0 / math.sqrt(float(case.d_k))
    scale_bf16 = float_to_bf16_bits(scale_value)
    scale_float = bf16_bits_to_float(scale_bf16)

    ref_bits: list[int] = []
    for qr in range(case.q_rows):
        scores = []
        for kv in range(case.kv_rows):
            dot = 0.0
            for d in range(case.d_k):
                dot += q[qr * case.d_k + d] * k[kv * case.d_k + d]
            scores.append(dot * scale_float)
        row_max = max(scores)
        exps = [math.exp(score - row_max) for score in scores]
        denom = sum(exps)
        probs = [x / denom for x in exps]
        for vc in range(case.value_cols):
            acc = 0.0
            for kv in range(case.kv_rows):
                acc += probs[kv] * v[kv * case.value_cols + vc]
            ref_bits.append(float_to_bf16_bits(acc))

    return {
        "q": q_bits,
        "k": k_bits,
        "v": v_bits,
        "ref": ref_bits,
        "scale_bf16": scale_bf16,
    }


def make_case(case: AttentionCase):
    if torch is not None:
        return make_case_torch(case)
    return make_case_stdlib(case)


def write_assets(cases: tuple[AttentionCase, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "attention_operator_cases.h"
    source = out_dir / "attention_operator_cases.c"

    max_q_rows = max(case.q_rows for case in cases)
    max_kv_rows = max(case.kv_rows for case in cases)
    max_d_k = max(case.d_k for case in cases)
    max_value_cols = max(case.value_cols for case in cases)

    generated = [(case, make_case(case)) for case in cases]

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
    backend = "torch" if torch is not None else "stdlib"
    print(f"wrote attention operator cases to {args.out_dir} using {backend} backend")


if __name__ == "__main__":
    main()
