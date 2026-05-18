#!/usr/bin/env python3
"""Export full transformer-layer cases as linkable C assets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TransformerLayerCase:
    name: str
    seq_len: int
    d_model: int
    n_heads: int
    hidden_dim: int
    seed: int
    input_scale: float = 0.5
    weight_scale: float = 0.03125
    tolerance_x100000: int = 75000

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


DEFAULT_CASES = (
    TransformerLayerCase("layer_seq8_dm32_h4_ff64", 8, 32, 4, 64, 401),
    TransformerLayerCase("layer_seq16_dm64_h4_ff128", 16, 64, 4, 128, 402),
    TransformerLayerCase("layer_seq32_dm64_h4_ff128", 32, 64, 4, 128, 403),
    TransformerLayerCase("layer_seq32_dm128_h4_ff256", 32, 128, 4, 256, 404),
    TransformerLayerCase("layer_seq64_dm128_h4_ff256", 64, 128, 4, 256, 405),
    TransformerLayerCase("layer_seq128_dm256_h4_ff256", 128, 256, 4, 256, 406),
    TransformerLayerCase("layer_seq64_dm512_h8_ff512", 64, 512, 8, 512, 407, 0.25, 0.015625),
    TransformerLayerCase("layer_seq128_dm512_h8_ff512", 128, 512, 8, 512, 408, 0.25, 0.015625),
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


def uniform_scaled(generator: torch.Generator, rows: int, cols: int, scale: float) -> torch.Tensor:
    return (torch.rand((rows, cols), generator=generator, dtype=torch.float32) * 2.0 - 1.0) * scale


def layer_norm_bf16(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mean = x.float().mean(dim=-1, keepdim=True)
    var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
    y = (x.float() - mean) * torch.rsqrt(var + 1e-5)
    return (y * gamma.float() + beta.float()).to(torch.bfloat16)


def make_case(case: TransformerLayerCase) -> dict[str, list[int] | int]:
    if case.d_model % case.n_heads != 0:
        raise ValueError(f"{case.name}: d_model must be divisible by n_heads")
    if case.head_dim > 512 or case.hidden_dim > 512:
        raise ValueError(f"{case.name}: dimensions exceed current accelerator maxK=512")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    x = uniform_scaled(generator, case.seq_len, case.d_model, case.input_scale).to(torch.bfloat16)
    ln1_gamma = (1.0 + uniform_scaled(generator, 1, case.d_model, 0.05).reshape(-1)).to(torch.bfloat16)
    ln1_beta = uniform_scaled(generator, 1, case.d_model, 0.05).reshape(-1).to(torch.bfloat16)
    ln2_gamma = (1.0 + uniform_scaled(generator, 1, case.d_model, 0.05).reshape(-1)).to(torch.bfloat16)
    ln2_beta = uniform_scaled(generator, 1, case.d_model, 0.05).reshape(-1).to(torch.bfloat16)

    wq = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale).to(torch.bfloat16)
    wk = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale).to(torch.bfloat16)
    wv = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale).to(torch.bfloat16)
    wo = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale).to(torch.bfloat16)
    w1 = uniform_scaled(generator, case.d_model, case.hidden_dim, case.weight_scale).to(torch.bfloat16)
    w2 = uniform_scaled(generator, case.hidden_dim, case.d_model, case.weight_scale).to(torch.bfloat16)

    ln1 = layer_norm_bf16(x, ln1_gamma, ln1_beta)
    q = torch.matmul(ln1.float(), wq.float()).to(torch.bfloat16)
    k = torch.matmul(ln1.float(), wk.float()).to(torch.bfloat16)
    v = torch.matmul(ln1.float(), wv.float()).to(torch.bfloat16)

    scale_value = 1.0 / math.sqrt(float(case.head_dim))
    scale_bf16 = bf16_scalar_bits(scale_value)
    scale_float = bf16_scalar_float(scale_value)

    heads = []
    for h in range(case.n_heads):
        start = h * case.head_dim
        end = start + case.head_dim
        scores = torch.matmul(q[:, start:end].float(), k[:, start:end].float().transpose(0, 1)) * scale_float
        probs = torch.softmax(scores, dim=-1)
        heads.append(torch.matmul(probs, v[:, start:end].float()).to(torch.bfloat16))

    context = torch.cat(heads, dim=1).to(torch.bfloat16)
    attn_out = torch.matmul(context.float(), wo.float()).to(torch.bfloat16)
    res1 = (x.float() + attn_out.float()).to(torch.bfloat16)
    ln2 = layer_norm_bf16(res1, ln2_gamma, ln2_beta)
    ff1 = torch.matmul(ln2.float(), w1.float()).to(torch.bfloat16)
    act = torch.nn.functional.gelu(ff1.float(), approximate="none").to(torch.bfloat16)
    ff2 = torch.matmul(act.float(), w2.float()).to(torch.bfloat16)
    expected = (res1.float() + ff2.float()).to(torch.bfloat16)

    return {
        "x": bf16_bits(x),
        "ln1_gamma": bf16_bits(ln1_gamma),
        "ln1_beta": bf16_bits(ln1_beta),
        "ln2_gamma": bf16_bits(ln2_gamma),
        "ln2_beta": bf16_bits(ln2_beta),
        "wq": bf16_bits(wq),
        "wk": bf16_bits(wk),
        "wv": bf16_bits(wv),
        "wo": bf16_bits(wo),
        "w1": bf16_bits(w1),
        "w2": bf16_bits(w2),
        "expected": bf16_bits(expected),
        "scale_bf16": scale_bf16,
    }


def print_case_sample(case: TransformerLayerCase, data: dict[str, list[int] | int]) -> None:
    exp = data["expected"]
    mid_idx = (case.seq_len // 2) * case.d_model + (case.d_model // 2)
    print(
        "SAMPLE,"
        f"{case.name},seq={case.seq_len},d_model={case.d_model},heads={case.n_heads},"
        f"head_dim={case.head_dim},hidden={case.hidden_dim},scale_bf16=0x{data['scale_bf16']:04x},"
        f"x00={bf16_bit_to_float(data['x'][0]):.6f},ref00={bf16_bit_to_float(exp[0]):.6f},"
        f"ref_mid={bf16_bit_to_float(exp[mid_idx]):.6f},ref_last={bf16_bit_to_float(exp[-1]):.6f}"
    )


def write_assets(cases: tuple[TransformerLayerCase, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "transformer_layer_cases.h"
    source = out_dir / "transformer_layer_cases.c"
    generated = [(case, make_case(case)) for case in cases]

    max_seq_len = max(case.seq_len for case in cases)
    max_d_model = max(case.d_model for case in cases)
    max_heads = max(case.n_heads for case in cases)
    max_head_dim = max(case.head_dim for case in cases)
    max_hidden_dim = max(case.hidden_dim for case in cases)

    print("SAMPLE_HEADER,name,seq,d_model,heads,head_dim,hidden,scale_bf16,x00,ref00,ref_mid,ref_last")
    for case, data in generated:
        print_case_sample(case, data)

    with header.open("w", encoding="utf-8") as out:
        out.write("#ifndef GIRDAP_GENERATED_TRANSFORMER_LAYER_CASES_H\n")
        out.write("#define GIRDAP_GENERATED_TRANSFORMER_LAYER_CASES_H\n\n")
        out.write("#include <stdint.h>\n\n")
        out.write(f"#define TRANSFORMER_LAYER_MAX_SEQ_LEN {max_seq_len}\n")
        out.write(f"#define TRANSFORMER_LAYER_MAX_D_MODEL {max_d_model}\n")
        out.write(f"#define TRANSFORMER_LAYER_MAX_HEADS {max_heads}\n")
        out.write(f"#define TRANSFORMER_LAYER_MAX_HEAD_DIM {max_head_dim}\n")
        out.write(f"#define TRANSFORMER_LAYER_MAX_HIDDEN_DIM {max_hidden_dim}\n\n")
        out.write("typedef struct {\n")
        out.write("  const char *name;\n")
        out.write("  int seq_len;\n")
        out.write("  int d_model;\n")
        out.write("  int n_heads;\n")
        out.write("  int head_dim;\n")
        out.write("  int hidden_dim;\n")
        out.write("  uint16_t scale_bf16;\n")
        out.write("  uint32_t tolerance_x100000;\n")
        for field in ("x", "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "wq", "wk", "wv", "wo", "w1", "w2", "expected"):
            out.write(f"  const uint16_t *{field};\n")
        out.write("} transformer_layer_case_t;\n\n")
        out.write("extern const transformer_layer_case_t transformer_layer_cases[];\n")
        out.write("extern const int transformer_layer_case_count;\n\n")
        out.write("#endif\n")

    with source.open("w", encoding="utf-8") as out:
        out.write('#include "transformer_layer_cases.h"\n\n')
        fields = ("x", "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "wq", "wk", "wv", "wo", "w1", "w2", "expected")
        for idx, (case, data) in enumerate(generated):
            prefix = f"transformer_layer_case_{idx}"
            out.write(f"/* {case.name}: seq={case.seq_len}, d_model={case.d_model}, hidden={case.hidden_dim} */\n")
            for field in fields:
                emit_array(out, f"{prefix}_{field}", data[field])

        out.write("const transformer_layer_case_t transformer_layer_cases[] = {\n")
        for idx, (case, data) in enumerate(generated):
            prefix = f"transformer_layer_case_{idx}"
            out.write("  {\n")
            out.write(f'    .name = "{case.name}",\n')
            out.write(f"    .seq_len = {case.seq_len},\n")
            out.write(f"    .d_model = {case.d_model},\n")
            out.write(f"    .n_heads = {case.n_heads},\n")
            out.write(f"    .head_dim = {case.head_dim},\n")
            out.write(f"    .hidden_dim = {case.hidden_dim},\n")
            out.write(f"    .scale_bf16 = 0x{data['scale_bf16']:04x},\n")
            out.write(f"    .tolerance_x100000 = {case.tolerance_x100000},\n")
            for field in fields:
                out.write(f"    .{field} = {prefix}_{field},\n")
            out.write("  },\n")
        out.write("};\n\n")
        out.write("const int transformer_layer_case_count = sizeof(transformer_layer_cases) / sizeof(transformer_layer_cases[0]);\n")


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
    print(f"wrote transformer layer cases to {args.out_dir} using PyTorch backend")


if __name__ == "__main__":
    main()
