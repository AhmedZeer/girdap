#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026
"""Export GPT-2-style causal prefill cases as linkable C assets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Gpt2PrefillCase:
    name: str
    seq_len: int
    d_model: int
    n_heads: int
    hidden_dim: int
    n_layers: int
    vocab_size: int
    seed: int
    embedding_scale: float = 0.5
    weight_scale: float = 0.03125
    bias_scale: float = 0.02
    tolerance_x100000: int = 100000

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# Start with smaller ramp-up cases, then run the 512-oriented throughput cases.
DEFAULT_CASES = (
    Gpt2PrefillCase("gpt2prefill_l1_s16_d64_h4_ff128_vocab128", 16, 64, 4, 128, 1, 128, 801),
    Gpt2PrefillCase("gpt2prefill_l1_s32_d128_h4_ff256_vocab256", 32, 128, 4, 256, 1, 256, 802),
    Gpt2PrefillCase("gpt2prefill_l1_s64_d128_h4_ff256_vocab256", 64, 128, 4, 256, 1, 256, 803),
    # Gpt2PrefillCase("gpt2prefill_l4_s128_d128_h4_ff256_vocab256", 128, 128, 4, 256, 1, 256, 804),
    # Gpt2PrefillCase("gpt2prefill_l2_s128_d256_h4_ff256_vocab256", 128, 256, 4, 256, 1, 256, 805, 0.25, 0.015625),
    # Gpt2PrefillCase("gpt2prefill_l1_s256_d128_h4_ff256_vocab256", 256, 128, 4, 256, 1, 256, 806, 0.25, 0.015625),
    Gpt2PrefillCase("gpt2prefill_512_l1_s128_d512_h8_ff512_vocab512", 128, 512, 8, 512, 1, 512, 811, 0.25, 0.015625),
    Gpt2PrefillCase("gpt2prefill_512_l2_s128_d512_h8_ff512_vocab512", 128, 512, 8, 512, 2, 512, 812, 0.25, 0.015625),
    Gpt2PrefillCase("gpt2prefill_512_l1_s256_d512_h8_ff512_vocab512", 256, 512, 8, 512, 1, 512, 813, 0.25, 0.015625),
    Gpt2PrefillCase("gpt2prefill_512_l2_s256_d512_h8_ff512_vocab512", 256, 512, 8, 512, 2, 512, 814, 0.25, 0.015625),
    Gpt2PrefillCase("gpt2prefill_512_l1_s512_d256_h4_ff512_vocab512", 512, 256, 4, 512, 1, 512, 815, 0.25, 0.015625),
    Gpt2PrefillCase("gpt2prefill_512_l1_s512_d512_h8_ff512_vocab512", 512, 512, 8, 512, 1, 512, 816, 0.20, 0.01171875),
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


def uniform_scaled(generator: torch.Generator, *shape: int, scale: float) -> torch.Tensor:
    return (torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0) * scale


def layer_norm_bf16(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mean = x.float().mean(dim=-1, keepdim=True)
    var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
    y = (x.float() - mean) * torch.rsqrt(var + 1e-5)
    return (y * gamma.float() + beta.float()).to(torch.bfloat16)


def linear_bf16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    y = torch.matmul(x.float(), weight.float())
    if bias is not None:
        y = y + bias.float()
    return y.to(torch.bfloat16)


def emit_u16_array(out, name: str, values: list[int]) -> None:
    out.write(f"const uint16_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 12):
        out.write("  " + ", ".join(f"0x{x:04x}" for x in values[i : i + 12]) + ",\n")
    out.write("};\n\n")


def emit_i32_array(out, name: str, values: list[int]) -> None:
    out.write(f"const int32_t {name}[] __attribute__((aligned(64))) = {{\n")
    for i in range(0, len(values), 16):
        out.write("  " + ", ".join(str(x) for x in values[i : i + 16]) + ",\n")
    out.write("};\n\n")


def flatten(params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params]).contiguous()


def make_case(case: Gpt2PrefillCase) -> dict[str, list[int] | int]:
    if case.d_model % case.n_heads != 0:
        raise ValueError(f"{case.name}: d_model must be divisible by n_heads")
    if case.seq_len > 512 or case.d_model > 512 or case.hidden_dim > 512 or case.head_dim > 512:
        raise ValueError(f"{case.name}: dimensions exceed current maxK=512 AGFI envelope")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    token_ids = torch.randint(0, case.vocab_size, (case.seq_len,), generator=generator, dtype=torch.int32)
    token_embeddings = uniform_scaled(generator, case.vocab_size, case.d_model, scale=case.embedding_scale).to(torch.bfloat16)
    position_embeddings = uniform_scaled(generator, case.seq_len, case.d_model, scale=case.embedding_scale).to(torch.bfloat16)

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
    bq = []
    bk = []
    bv = []
    bo = []
    b1 = []
    b2 = []

    x = (
        token_embeddings[token_ids.long()].float()
        + position_embeddings[torch.arange(case.seq_len)].float()
    ).to(torch.bfloat16)

    scale_value = 1.0 / math.sqrt(float(case.head_dim))
    scale_bf16 = bf16_scalar_bits(scale_value)
    scale_float = bf16_scalar_float(scale_value)
    causal_mask = torch.triu(
        torch.ones((case.seq_len, case.seq_len), dtype=torch.bool),
        diagonal=1,
    )

    for _ in range(case.n_layers):
        g1 = (1.0 + uniform_scaled(generator, case.d_model, scale=0.05)).to(torch.bfloat16)
        be1 = uniform_scaled(generator, case.d_model, scale=0.05).to(torch.bfloat16)
        g2 = (1.0 + uniform_scaled(generator, case.d_model, scale=0.05)).to(torch.bfloat16)
        be2 = uniform_scaled(generator, case.d_model, scale=0.05).to(torch.bfloat16)

        qw = uniform_scaled(generator, case.d_model, case.d_model, scale=case.weight_scale).to(torch.bfloat16)
        kw = uniform_scaled(generator, case.d_model, case.d_model, scale=case.weight_scale).to(torch.bfloat16)
        vw = uniform_scaled(generator, case.d_model, case.d_model, scale=case.weight_scale).to(torch.bfloat16)
        ow = uniform_scaled(generator, case.d_model, case.d_model, scale=case.weight_scale).to(torch.bfloat16)
        f1 = uniform_scaled(generator, case.d_model, case.hidden_dim, scale=case.weight_scale).to(torch.bfloat16)
        f2 = uniform_scaled(generator, case.hidden_dim, case.d_model, scale=case.weight_scale).to(torch.bfloat16)

        qb = uniform_scaled(generator, case.d_model, scale=case.bias_scale).to(torch.bfloat16)
        kb = uniform_scaled(generator, case.d_model, scale=case.bias_scale).to(torch.bfloat16)
        vb = uniform_scaled(generator, case.d_model, scale=case.bias_scale).to(torch.bfloat16)
        ob = uniform_scaled(generator, case.d_model, scale=case.bias_scale).to(torch.bfloat16)
        f1b = uniform_scaled(generator, case.hidden_dim, scale=case.bias_scale).to(torch.bfloat16)
        f2b = uniform_scaled(generator, case.d_model, scale=case.bias_scale).to(torch.bfloat16)

        ln1_gamma.append(g1)
        ln1_beta.append(be1)
        ln2_gamma.append(g2)
        ln2_beta.append(be2)
        wq.append(qw)
        wk.append(kw)
        wv.append(vw)
        wo.append(ow)
        w1.append(f1)
        w2.append(f2)
        bq.append(qb)
        bk.append(kb)
        bv.append(vb)
        bo.append(ob)
        b1.append(f1b)
        b2.append(f2b)

        ln1 = layer_norm_bf16(x, g1, be1)
        q = linear_bf16(ln1, qw, qb)
        k = linear_bf16(ln1, kw, kb)
        v = linear_bf16(ln1, vw, vb)

        heads = []
        for h in range(case.n_heads):
            start = h * case.head_dim
            end = start + case.head_dim
            scores = torch.matmul(q[:, start:end].float(), k[:, start:end].float().transpose(0, 1)) * scale_float
            scores = scores.masked_fill(causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(probs, v[:, start:end].float()).to(torch.bfloat16))
        context = torch.cat(heads, dim=1).to(torch.bfloat16)
        attn_out = linear_bf16(context, ow, ob)
        x = (x.float() + attn_out.float()).to(torch.bfloat16)

        ln2 = layer_norm_bf16(x, g2, be2)
        hidden = linear_bf16(ln2, f1, f1b)
        act = torch.nn.functional.gelu(hidden.float(), approximate="none").to(torch.bfloat16)
        ffn = linear_bf16(act, f2, f2b)
        x = (x.float() + ffn.float()).to(torch.bfloat16)

    final_ln_gamma = (1.0 + uniform_scaled(generator, case.d_model, scale=0.05)).to(torch.bfloat16)
    final_ln_beta = uniform_scaled(generator, case.d_model, scale=0.05).to(torch.bfloat16)
    lm_head = uniform_scaled(generator, case.d_model, case.vocab_size, scale=case.weight_scale).to(torch.bfloat16)
    final_hidden = layer_norm_bf16(x, final_ln_gamma, final_ln_beta)
    expected_logits = linear_bf16(final_hidden, lm_head, None)

    return {
        "token_ids": token_ids.tolist(),
        "token_embeddings": bf16_bits(token_embeddings),
        "position_embeddings": bf16_bits(position_embeddings),
        "ln1_gamma": bf16_bits(flatten(ln1_gamma)),
        "ln1_beta": bf16_bits(flatten(ln1_beta)),
        "ln2_gamma": bf16_bits(flatten(ln2_gamma)),
        "ln2_beta": bf16_bits(flatten(ln2_beta)),
        "wq": bf16_bits(flatten(wq)),
        "wk": bf16_bits(flatten(wk)),
        "wv": bf16_bits(flatten(wv)),
        "wo": bf16_bits(flatten(wo)),
        "w1": bf16_bits(flatten(w1)),
        "w2": bf16_bits(flatten(w2)),
        "bq": bf16_bits(flatten(bq)),
        "bk": bf16_bits(flatten(bk)),
        "bv": bf16_bits(flatten(bv)),
        "bo": bf16_bits(flatten(bo)),
        "b1": bf16_bits(flatten(b1)),
        "b2": bf16_bits(flatten(b2)),
        "final_ln_gamma": bf16_bits(final_ln_gamma),
        "final_ln_beta": bf16_bits(final_ln_beta),
        "lm_head": bf16_bits(lm_head),
        "expected_logits": bf16_bits(expected_logits),
        "scale_bf16": scale_bf16,
    }


def print_case_sample(case: Gpt2PrefillCase, data: dict[str, list[int] | int]) -> None:
    logits = data["expected_logits"]
    mid_idx = (case.seq_len // 2) * case.vocab_size + (case.vocab_size // 2)
    print(
        "SAMPLE,"
        f"{case.name},seq={case.seq_len},d_model={case.d_model},heads={case.n_heads},"
        f"head_dim={case.head_dim},hidden={case.hidden_dim},layers={case.n_layers},"
        f"vocab={case.vocab_size},scale_bf16=0x{data['scale_bf16']:04x},"
        f"tok0={data['token_ids'][0]},logit00={bf16_bit_to_float(logits[0]):.6f},"
        f"logit_mid={bf16_bit_to_float(logits[mid_idx]):.6f},logit_last={bf16_bit_to_float(logits[-1]):.6f}"
    )


def write_assets(cases: tuple[Gpt2PrefillCase, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "gpt2_prefill_cases.h"
    source = out_dir / "gpt2_prefill_cases.c"
    generated = [(case, make_case(case)) for case in cases]

    max_seq_len = max(case.seq_len for case in cases)
    max_d_model = max(case.d_model for case in cases)
    max_heads = max(case.n_heads for case in cases)
    max_head_dim = max(case.head_dim for case in cases)
    max_hidden_dim = max(case.hidden_dim for case in cases)
    max_layers = max(case.n_layers for case in cases)
    max_vocab_size = max(case.vocab_size for case in cases)
    max_gemm_n = max(max_d_model, max_hidden_dim, max_vocab_size)
    max_gemm_k = max(max_d_model, max_hidden_dim)

    print("SAMPLE_HEADER,name,seq,d_model,heads,head_dim,hidden,layers,vocab,scale_bf16,tok0,logit00,logit_mid,logit_last")
    for case, data in generated:
        print_case_sample(case, data)

    with header.open("w", encoding="utf-8") as out:
        out.write("#ifndef GIRDAP_GENERATED_GPT2_PREFILL_CASES_H\n")
        out.write("#define GIRDAP_GENERATED_GPT2_PREFILL_CASES_H\n\n")
        out.write("#include <stdint.h>\n\n")
        out.write(f"#define GPT2_PREFILL_MAX_SEQ_LEN {max_seq_len}\n")
        out.write(f"#define GPT2_PREFILL_MAX_D_MODEL {max_d_model}\n")
        out.write(f"#define GPT2_PREFILL_MAX_HEADS {max_heads}\n")
        out.write(f"#define GPT2_PREFILL_MAX_HEAD_DIM {max_head_dim}\n")
        out.write(f"#define GPT2_PREFILL_MAX_HIDDEN_DIM {max_hidden_dim}\n")
        out.write(f"#define GPT2_PREFILL_MAX_LAYERS {max_layers}\n")
        out.write(f"#define GPT2_PREFILL_MAX_VOCAB_SIZE {max_vocab_size}\n")
        out.write(f"#define GPT2_PREFILL_MAX_GEMM_N {max_gemm_n}\n")
        out.write(f"#define GPT2_PREFILL_MAX_GEMM_K {max_gemm_k}\n\n")
        out.write("typedef struct {\n")
        out.write("  const char *name;\n")
        out.write("  int seq_len;\n")
        out.write("  int d_model;\n")
        out.write("  int n_heads;\n")
        out.write("  int head_dim;\n")
        out.write("  int hidden_dim;\n")
        out.write("  int n_layers;\n")
        out.write("  int vocab_size;\n")
        out.write("  uint16_t scale_bf16;\n")
        out.write("  uint32_t tolerance_x100000;\n")
        out.write("  const int32_t *token_ids;\n")
        for field in (
            "token_embeddings", "position_embeddings", "ln1_gamma", "ln1_beta",
            "ln2_gamma", "ln2_beta", "wq", "wk", "wv", "wo", "w1", "w2",
            "bq", "bk", "bv", "bo", "b1", "b2", "final_ln_gamma",
            "final_ln_beta", "lm_head", "expected_logits",
        ):
            out.write(f"  const uint16_t *{field};\n")
        out.write("} gpt2_prefill_case_t;\n\n")
        out.write("extern const gpt2_prefill_case_t gpt2_prefill_cases[];\n")
        out.write("extern const int gpt2_prefill_case_count;\n\n")
        out.write("#endif\n")

    fields = (
        "token_embeddings", "position_embeddings", "ln1_gamma", "ln1_beta",
        "ln2_gamma", "ln2_beta", "wq", "wk", "wv", "wo", "w1", "w2",
        "bq", "bk", "bv", "bo", "b1", "b2", "final_ln_gamma",
        "final_ln_beta", "lm_head", "expected_logits",
    )
    with source.open("w", encoding="utf-8") as out:
        out.write('#include "gpt2_prefill_cases.h"\n\n')
        for idx, (case, data) in enumerate(generated):
            prefix = f"gpt2_prefill_case_{idx}"
            out.write(f"/* {case.name}: seq={case.seq_len}, d_model={case.d_model}, layers={case.n_layers} */\n")
            emit_i32_array(out, f"{prefix}_token_ids", data["token_ids"])
            for field in fields:
                emit_u16_array(out, f"{prefix}_{field}", data[field])

        out.write("const gpt2_prefill_case_t gpt2_prefill_cases[] = {\n")
        for idx, (case, data) in enumerate(generated):
            prefix = f"gpt2_prefill_case_{idx}"
            out.write("  {\n")
            out.write(f'    .name = "{case.name}",\n')
            out.write(f"    .seq_len = {case.seq_len},\n")
            out.write(f"    .d_model = {case.d_model},\n")
            out.write(f"    .n_heads = {case.n_heads},\n")
            out.write(f"    .head_dim = {case.head_dim},\n")
            out.write(f"    .hidden_dim = {case.hidden_dim},\n")
            out.write(f"    .n_layers = {case.n_layers},\n")
            out.write(f"    .vocab_size = {case.vocab_size},\n")
            out.write(f"    .scale_bf16 = 0x{data['scale_bf16']:04x},\n")
            out.write(f"    .tolerance_x100000 = {case.tolerance_x100000},\n")
            out.write(f"    .token_ids = {prefix}_token_ids,\n")
            for field in fields:
                out.write(f"    .{field} = {prefix}_{field},\n")
            out.write("  },\n")
        out.write("};\n\n")
        out.write("const int gpt2_prefill_case_count = sizeof(gpt2_prefill_cases) / sizeof(gpt2_prefill_cases[0]);\n")


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
    print(f"wrote GPT-2 prefill cases to {args.out_dir} using PyTorch backend")


if __name__ == "__main__":
    main()
