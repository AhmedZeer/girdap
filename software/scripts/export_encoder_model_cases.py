#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026
"""Export BERT/ViT-style encoder model cases as linkable C assets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class EncoderModelCase:
    name: str
    seq_len: int
    d_model: int
    n_heads: int
    hidden_dim: int
    n_layers: int
    seed: int
    input_scale: float = 0.5
    weight_scale: float = 0.03125
    tolerance_x100000: int = 100000

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


BERT_CASES = (
    EncoderModelCase("bert_tiny_l1_s64_d128_h4_ff256", 64, 128, 4, 256, 1, 501),
    EncoderModelCase("bert_tiny_l2_s64_d128_h4_ff256", 64, 128, 4, 256, 2, 502),
    EncoderModelCase("bert_small_l2_s128_d256_h4_ff256", 128, 256, 4, 256, 2, 503),
    EncoderModelCase("bert_max_l4_s256_d256_h4_ff256", 256, 256, 4, 256, 4, 504),
)

VIT_CASES = (
    EncoderModelCase("vit_tiny_l1_s65_d128_h4_ff256", 65, 128, 4, 256, 1, 601),
    EncoderModelCase("vit_tiny_l2_s65_d128_h4_ff256", 65, 128, 4, 256, 2, 602),
    EncoderModelCase("vit_patch_l2_s197_d256_h4_ff256", 197, 256, 4, 256, 2, 603),
    EncoderModelCase("vit_max_l4_s197_d256_h4_ff256", 197, 256, 4, 256, 4, 604),
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


def uniform_scaled(generator: torch.Generator, rows: int, cols: int, scale: float) -> torch.Tensor:
    return (torch.rand((rows, cols), generator=generator, dtype=torch.float32) * 2.0 - 1.0) * scale


def layer_norm_bf16(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mean = x.float().mean(dim=-1, keepdim=True)
    var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
    y = (x.float() - mean) * torch.rsqrt(var + 1e-5)
    return (y * gamma.float() + beta.float()).to(torch.bfloat16)


def emit_array(out, name: str, values: list[int]) -> None:
    out.write(f"const uint16_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 12):
        out.write("  " + ", ".join(f"0x{x:04x}" for x in values[i : i + 12]) + ",\n")
    out.write("};\n\n")


def flat_layer_params(params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params]).contiguous()


def make_case(case: EncoderModelCase) -> dict[str, list[int] | int]:
    if case.d_model % case.n_heads != 0:
        raise ValueError(f"{case.name}: d_model must divide n_heads")
    if case.head_dim > 256 or case.hidden_dim > 256 or case.d_model > 256:
        raise ValueError(f"{case.name}: dimensions exceed current RTL maxK=256")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    x = uniform_scaled(generator, case.seq_len, case.d_model, case.input_scale).to(torch.bfloat16)
    scale_value = 1.0 / math.sqrt(float(case.head_dim))
    scale_bf16 = bf16_scalar_bits(scale_value)
    scale_float = bf16_scalar_float(scale_value)

    ln1_gamma = []
    ln1_beta = []
    ln2_gamma = []
    ln2_beta = []
    wq = []
    wk = []
    wv = []
    wo = []
    w1 = []
    w2 = []

    y = x
    for _ in range(case.n_layers):
        l1g = (1.0 + uniform_scaled(generator, 1, case.d_model, 0.05).reshape(-1)).to(torch.bfloat16)
        l1b = uniform_scaled(generator, 1, case.d_model, 0.05).reshape(-1).to(torch.bfloat16)
        l2g = (1.0 + uniform_scaled(generator, 1, case.d_model, 0.05).reshape(-1)).to(torch.bfloat16)
        l2b = uniform_scaled(generator, 1, case.d_model, 0.05).reshape(-1).to(torch.bfloat16)
        qw = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale).to(torch.bfloat16)
        kw = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale).to(torch.bfloat16)
        vw = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale).to(torch.bfloat16)
        ow = uniform_scaled(generator, case.d_model, case.d_model, case.weight_scale).to(torch.bfloat16)
        f1 = uniform_scaled(generator, case.d_model, case.hidden_dim, case.weight_scale).to(torch.bfloat16)
        f2 = uniform_scaled(generator, case.hidden_dim, case.d_model, case.weight_scale).to(torch.bfloat16)

        ln1_gamma.append(l1g)
        ln1_beta.append(l1b)
        ln2_gamma.append(l2g)
        ln2_beta.append(l2b)
        wq.append(qw)
        wk.append(kw)
        wv.append(vw)
        wo.append(ow)
        w1.append(f1)
        w2.append(f2)

        ln1 = layer_norm_bf16(y, l1g, l1b)
        q = torch.matmul(ln1.float(), qw.float()).to(torch.bfloat16)
        k = torch.matmul(ln1.float(), kw.float()).to(torch.bfloat16)
        v = torch.matmul(ln1.float(), vw.float()).to(torch.bfloat16)

        heads = []
        for h in range(case.n_heads):
            start = h * case.head_dim
            end = start + case.head_dim
            scores = torch.matmul(q[:, start:end].float(), k[:, start:end].float().transpose(0, 1)) * scale_float
            probs = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(probs, v[:, start:end].float()).to(torch.bfloat16))
        context = torch.cat(heads, dim=1).to(torch.bfloat16)
        attn_out = torch.matmul(context.float(), ow.float()).to(torch.bfloat16)
        res1 = (y.float() + attn_out.float()).to(torch.bfloat16)
        ln2 = layer_norm_bf16(res1, l2g, l2b)
        hidden = torch.matmul(ln2.float(), f1.float()).to(torch.bfloat16)
        act = torch.nn.functional.gelu(hidden.float(), approximate="none").to(torch.bfloat16)
        mlp = torch.matmul(act.float(), f2.float()).to(torch.bfloat16)
        y = (res1.float() + mlp.float()).to(torch.bfloat16)

    return {
        "x": bf16_bits(x),
        "ln1_gamma": bf16_bits(flat_layer_params(ln1_gamma)),
        "ln1_beta": bf16_bits(flat_layer_params(ln1_beta)),
        "ln2_gamma": bf16_bits(flat_layer_params(ln2_gamma)),
        "ln2_beta": bf16_bits(flat_layer_params(ln2_beta)),
        "wq": bf16_bits(flat_layer_params(wq)),
        "wk": bf16_bits(flat_layer_params(wk)),
        "wv": bf16_bits(flat_layer_params(wv)),
        "wo": bf16_bits(flat_layer_params(wo)),
        "w1": bf16_bits(flat_layer_params(w1)),
        "w2": bf16_bits(flat_layer_params(w2)),
        "expected": bf16_bits(y),
        "scale_bf16": scale_bf16,
    }


def print_case_sample(case: EncoderModelCase, data: dict[str, list[int] | int]) -> None:
    exp = data["expected"]
    mid_idx = (case.seq_len // 2) * case.d_model + (case.d_model // 2)
    print(
        "SAMPLE,"
        f"{case.name},layers={case.n_layers},seq={case.seq_len},d_model={case.d_model},"
        f"heads={case.n_heads},head_dim={case.head_dim},hidden={case.hidden_dim},"
        f"scale_bf16=0x{data['scale_bf16']:04x},"
        f"x00={bf16_bit_to_float(data['x'][0]):.6f},"
        f"ref00={bf16_bit_to_float(exp[0]):.6f},"
        f"ref_mid={bf16_bit_to_float(exp[mid_idx]):.6f},"
        f"ref_last={bf16_bit_to_float(exp[-1]):.6f}"
    )


def write_assets(cases: tuple[EncoderModelCase, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "encoder_model_cases.h"
    source = out_dir / "encoder_model_cases.c"
    generated = [(case, make_case(case)) for case in cases]

    max_layers = max(case.n_layers for case in cases)
    max_seq_len = max(case.seq_len for case in cases)
    max_d_model = max(case.d_model for case in cases)
    max_heads = max(case.n_heads for case in cases)
    max_head_dim = max(case.head_dim for case in cases)
    max_hidden_dim = max(case.hidden_dim for case in cases)

    print("SAMPLE_HEADER,name,layers,seq,d_model,heads,head_dim,hidden,scale_bf16,x00,ref00,ref_mid,ref_last")
    for case, data in generated:
        print_case_sample(case, data)

    with header.open("w", encoding="utf-8") as out:
        out.write("#ifndef TOYROCC_GENERATED_ENCODER_MODEL_CASES_H\n")
        out.write("#define TOYROCC_GENERATED_ENCODER_MODEL_CASES_H\n\n")
        out.write("#include <stdint.h>\n\n")
        out.write(f"#define ENCODER_MODEL_MAX_LAYERS {max_layers}\n")
        out.write(f"#define ENCODER_MODEL_MAX_SEQ_LEN {max_seq_len}\n")
        out.write(f"#define ENCODER_MODEL_MAX_D_MODEL {max_d_model}\n")
        out.write(f"#define ENCODER_MODEL_MAX_HEADS {max_heads}\n")
        out.write(f"#define ENCODER_MODEL_MAX_HEAD_DIM {max_head_dim}\n")
        out.write(f"#define ENCODER_MODEL_MAX_HIDDEN_DIM {max_hidden_dim}\n\n")
        out.write("typedef struct {\n")
        out.write("  const char *name;\n")
        out.write("  int seq_len;\n")
        out.write("  int d_model;\n")
        out.write("  int n_heads;\n")
        out.write("  int head_dim;\n")
        out.write("  int hidden_dim;\n")
        out.write("  int n_layers;\n")
        out.write("  uint16_t scale_bf16;\n")
        out.write("  uint32_t tolerance_x100000;\n")
        for field in ("x", "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "wq", "wk", "wv", "wo", "w1", "w2", "expected"):
            out.write(f"  const uint16_t *{field};\n")
        out.write("} encoder_model_case_t;\n\n")
        out.write("extern const encoder_model_case_t encoder_model_cases[];\n")
        out.write("extern const int encoder_model_case_count;\n\n")
        out.write("#endif\n")

    fields = ("x", "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta", "wq", "wk", "wv", "wo", "w1", "w2", "expected")
    with source.open("w", encoding="utf-8") as out:
        out.write('#include "encoder_model_cases.h"\n\n')
        for idx, (case, data) in enumerate(generated):
            prefix = f"encoder_model_case_{idx}"
            out.write(f"/* {case.name}: layers={case.n_layers}, seq={case.seq_len}, d_model={case.d_model} */\n")
            for field in fields:
                emit_array(out, f"{prefix}_{field}", data[field])

        out.write("const encoder_model_case_t encoder_model_cases[] = {\n")
        for idx, (case, data) in enumerate(generated):
            prefix = f"encoder_model_case_{idx}"
            out.write("  {\n")
            out.write(f'    .name = "{case.name}",\n')
            out.write(f"    .seq_len = {case.seq_len},\n")
            out.write(f"    .d_model = {case.d_model},\n")
            out.write(f"    .n_heads = {case.n_heads},\n")
            out.write(f"    .head_dim = {case.head_dim},\n")
            out.write(f"    .hidden_dim = {case.hidden_dim},\n")
            out.write(f"    .n_layers = {case.n_layers},\n")
            out.write(f"    .scale_bf16 = 0x{data['scale_bf16']:04x},\n")
            out.write(f"    .tolerance_x100000 = {case.tolerance_x100000},\n")
            for field in fields:
                out.write(f"    .{field} = {prefix}_{field},\n")
            out.write("  },\n")
        out.write("};\n\n")
        out.write("const int encoder_model_case_count = sizeof(encoder_model_cases) / sizeof(encoder_model_cases[0]);\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("bert", "vit"), required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "generated",
    )
    args = parser.parse_args()
    cases = BERT_CASES if args.suite == "bert" else VIT_CASES
    write_assets(cases, args.out_dir)
    print(f"wrote {args.suite} encoder model cases to {args.out_dir} using PyTorch backend")


if __name__ == "__main__":
    main()
