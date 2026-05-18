#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026
"""Export tiny-BERT inference cases as linkable C assets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TinyBertCase:
    name: str
    seq_len: int
    d_model: int
    n_heads: int
    hidden_dim: int
    n_layers: int
    seed: int
    vocab_size: int = 512
    type_vocab_size: int = 2
    num_classes: int = 2
    embedding_scale: float = 0.5
    weight_scale: float = 0.03125
    bias_scale: float = 0.02
    tolerance_x100000: int = 100000

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


DEFAULT_CASES = (
    TinyBertCase("tinybert_l1_s64_d128_h4_ff256_cls2", 32, 64, 4, 256, 1, 702),
    TinyBertCase("tinybert_l1_s128_d128_h4_ff256_cls2", 64, 128, 4, 256, 1, 703),
    TinyBertCase("tinybert_l1_s128_d128_h4_ff256_cls2", 128, 128, 4, 256, 1, 703),
    TinyBertCase("tinybert_l1_s128_d128_h4_ff256_cls2", 256, 256, 4, 256, 1, 703),
    TinyBertCase("tinybert_l1_s128_d128_h4_ff256_cls2", 512, 256, 4, 256, 1, 703),
    TinyBertCase("tinybert_512_l1_s512_d512_h8_ff512_cls2", 512, 512, 8, 512, 1, 711, embedding_scale=0.20, weight_scale=0.01171875),
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


def linear_bf16(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return (torch.matmul(x.float(), weight.float()) + bias.float()).to(torch.bfloat16)


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


def flat_layer_params(params: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params]).contiguous()


def make_case(case: TinyBertCase) -> dict[str, list[int] | int]:
    if case.d_model % case.n_heads != 0:
        raise ValueError(f"{case.name}: d_model must be divisible by n_heads")
    if case.d_model > 512 or case.hidden_dim > 512 or case.head_dim > 512:
        raise ValueError(f"{case.name}: dimensions exceed current RTL maxK=512")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    token_ids = torch.randint(0, case.vocab_size, (case.seq_len,), generator=generator, dtype=torch.int32)
    type_ids = torch.randint(0, case.type_vocab_size, (case.seq_len,), generator=generator, dtype=torch.int32)
    attention_mask = torch.ones((case.seq_len,), dtype=torch.int32)

    word_embeddings = uniform_scaled(generator, case.vocab_size, case.d_model, scale=case.embedding_scale).to(torch.bfloat16)
    position_embeddings = uniform_scaled(generator, case.seq_len, case.d_model, scale=case.embedding_scale).to(torch.bfloat16)
    token_type_embeddings = uniform_scaled(generator, case.type_vocab_size, case.d_model, scale=case.embedding_scale).to(torch.bfloat16)
    emb_ln_gamma = (1.0 + uniform_scaled(generator, case.d_model, scale=0.05)).to(torch.bfloat16)
    emb_ln_beta = uniform_scaled(generator, case.d_model, scale=0.05).to(torch.bfloat16)

    attn_ln_gamma = []
    attn_ln_beta = []
    ffn_ln_gamma = []
    ffn_ln_beta = []
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

    emb = (
        word_embeddings[token_ids.long()].float()
        + position_embeddings[torch.arange(case.seq_len)].float()
        + token_type_embeddings[type_ids.long()].float()
    ).to(torch.bfloat16)
    x = layer_norm_bf16(emb, emb_ln_gamma, emb_ln_beta)

    scale_value = 1.0 / math.sqrt(float(case.head_dim))
    scale_bf16 = bf16_scalar_bits(scale_value)
    scale_float = bf16_scalar_float(scale_value)

    for _ in range(case.n_layers):
        alg = (1.0 + uniform_scaled(generator, case.d_model, scale=0.05)).to(torch.bfloat16)
        alb = uniform_scaled(generator, case.d_model, scale=0.05).to(torch.bfloat16)
        flg = (1.0 + uniform_scaled(generator, case.d_model, scale=0.05)).to(torch.bfloat16)
        flb = uniform_scaled(generator, case.d_model, scale=0.05).to(torch.bfloat16)

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

        attn_ln_gamma.append(alg)
        attn_ln_beta.append(alb)
        ffn_ln_gamma.append(flg)
        ffn_ln_beta.append(flb)
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

        q = linear_bf16(x, qw, qb)
        k = linear_bf16(x, kw, kb)
        v = linear_bf16(x, vw, vb)

        heads = []
        for h in range(case.n_heads):
            start = h * case.head_dim
            end = start + case.head_dim
            scores = torch.matmul(q[:, start:end].float(), k[:, start:end].float().transpose(0, 1)) * scale_float
            scores = scores.masked_fill(attention_mask.view(1, -1) == 0, -10000.0)
            probs = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(probs, v[:, start:end].float()).to(torch.bfloat16))
        context = torch.cat(heads, dim=1).to(torch.bfloat16)

        attn_out = linear_bf16(context, ow, ob)
        x = layer_norm_bf16((x.float() + attn_out.float()).to(torch.bfloat16), alg, alb)
        hidden = linear_bf16(x, f1, f1b)
        act = torch.nn.functional.gelu(hidden.float(), approximate="none").to(torch.bfloat16)
        ffn_out = linear_bf16(act, f2, f2b)
        x = layer_norm_bf16((x.float() + ffn_out.float()).to(torch.bfloat16), flg, flb)

    pool_w = uniform_scaled(generator, case.d_model, case.d_model, scale=case.weight_scale).to(torch.bfloat16)
    pool_b = uniform_scaled(generator, case.d_model, scale=case.bias_scale).to(torch.bfloat16)
    classifier_w = uniform_scaled(generator, case.d_model, case.num_classes, scale=case.weight_scale).to(torch.bfloat16)
    classifier_b = uniform_scaled(generator, case.num_classes, scale=case.bias_scale).to(torch.bfloat16)
    pooled = torch.tanh(linear_bf16(x[0:1], pool_w, pool_b).float()).to(torch.bfloat16)
    logits = linear_bf16(pooled, classifier_w, classifier_b).reshape(-1)

    return {
        "token_ids": token_ids.tolist(),
        "type_ids": type_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "word_embeddings": bf16_bits(word_embeddings),
        "position_embeddings": bf16_bits(position_embeddings),
        "token_type_embeddings": bf16_bits(token_type_embeddings),
        "emb_ln_gamma": bf16_bits(emb_ln_gamma),
        "emb_ln_beta": bf16_bits(emb_ln_beta),
        "attn_ln_gamma": bf16_bits(flat_layer_params(attn_ln_gamma)),
        "attn_ln_beta": bf16_bits(flat_layer_params(attn_ln_beta)),
        "ffn_ln_gamma": bf16_bits(flat_layer_params(ffn_ln_gamma)),
        "ffn_ln_beta": bf16_bits(flat_layer_params(ffn_ln_beta)),
        "wq": bf16_bits(flat_layer_params(wq)),
        "wk": bf16_bits(flat_layer_params(wk)),
        "wv": bf16_bits(flat_layer_params(wv)),
        "wo": bf16_bits(flat_layer_params(wo)),
        "w1": bf16_bits(flat_layer_params(w1)),
        "w2": bf16_bits(flat_layer_params(w2)),
        "bq": bf16_bits(flat_layer_params(bq)),
        "bk": bf16_bits(flat_layer_params(bk)),
        "bv": bf16_bits(flat_layer_params(bv)),
        "bo": bf16_bits(flat_layer_params(bo)),
        "b1": bf16_bits(flat_layer_params(b1)),
        "b2": bf16_bits(flat_layer_params(b2)),
        "pool_w": bf16_bits(pool_w),
        "pool_b": bf16_bits(pool_b),
        "classifier_w": bf16_bits(classifier_w),
        "classifier_b": bf16_bits(classifier_b),
        "expected_logits": bf16_bits(logits),
        "scale_bf16": scale_bf16,
    }


def print_case_sample(case: TinyBertCase, data: dict[str, list[int] | int]) -> None:
    logits = data["expected_logits"]
    print(
        "SAMPLE,"
        f"{case.name},layers={case.n_layers},seq={case.seq_len},d_model={case.d_model},"
        f"heads={case.n_heads},head_dim={case.head_dim},hidden={case.hidden_dim},"
        f"vocab={case.vocab_size},classes={case.num_classes},"
        f"token0={data['token_ids'][0]},type0={data['type_ids'][0]},"
        f"logit0={bf16_bit_to_float(logits[0]):.6f},"
        f"logit_last={bf16_bit_to_float(logits[-1]):.6f}"
    )


def write_assets(cases: tuple[TinyBertCase, ...], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / "tiny_bert_cases.h"
    source = out_dir / "tiny_bert_cases.c"
    generated = [(case, make_case(case)) for case in cases]

    max_layers = max(case.n_layers for case in cases)
    max_seq_len = max(case.seq_len for case in cases)
    max_d_model = max(case.d_model for case in cases)
    max_heads = max(case.n_heads for case in cases)
    max_head_dim = max(case.head_dim for case in cases)
    max_hidden_dim = max(case.hidden_dim for case in cases)
    max_vocab_size = max(case.vocab_size for case in cases)
    max_type_vocab_size = max(case.type_vocab_size for case in cases)
    max_num_classes = max(case.num_classes for case in cases)

    print("SAMPLE_HEADER,name,layers,seq,d_model,heads,head_dim,hidden,vocab,classes,token0,type0,logit0,logit_last")
    for case, data in generated:
        print_case_sample(case, data)

    with header.open("w", encoding="utf-8") as out:
        out.write("#ifndef GIRDAP_GENERATED_TINY_BERT_CASES_H\n")
        out.write("#define GIRDAP_GENERATED_TINY_BERT_CASES_H\n\n")
        out.write("#include <stdint.h>\n\n")
        out.write(f"#define TINY_BERT_MAX_LAYERS {max_layers}\n")
        out.write(f"#define TINY_BERT_MAX_SEQ_LEN {max_seq_len}\n")
        out.write(f"#define TINY_BERT_MAX_D_MODEL {max_d_model}\n")
        out.write(f"#define TINY_BERT_MAX_HEADS {max_heads}\n")
        out.write(f"#define TINY_BERT_MAX_HEAD_DIM {max_head_dim}\n")
        out.write(f"#define TINY_BERT_MAX_HIDDEN_DIM {max_hidden_dim}\n")
        out.write(f"#define TINY_BERT_MAX_VOCAB_SIZE {max_vocab_size}\n")
        out.write(f"#define TINY_BERT_MAX_TYPE_VOCAB_SIZE {max_type_vocab_size}\n")
        out.write(f"#define TINY_BERT_MAX_NUM_CLASSES {max_num_classes}\n\n")
        out.write("typedef struct {\n")
        out.write("  const char *name;\n")
        out.write("  int seq_len;\n")
        out.write("  int vocab_size;\n")
        out.write("  int type_vocab_size;\n")
        out.write("  int d_model;\n")
        out.write("  int n_heads;\n")
        out.write("  int head_dim;\n")
        out.write("  int hidden_dim;\n")
        out.write("  int n_layers;\n")
        out.write("  int num_classes;\n")
        out.write("  uint16_t scale_bf16;\n")
        out.write("  uint32_t tolerance_x100000;\n")
        for field in ("token_ids", "type_ids", "attention_mask"):
            out.write(f"  const int32_t *{field};\n")
        fields = (
            "word_embeddings", "position_embeddings", "token_type_embeddings",
            "emb_ln_gamma", "emb_ln_beta", "attn_ln_gamma", "attn_ln_beta",
            "ffn_ln_gamma", "ffn_ln_beta", "wq", "wk", "wv", "wo", "w1", "w2",
            "bq", "bk", "bv", "bo", "b1", "b2", "pool_w", "pool_b",
            "classifier_w", "classifier_b", "expected_logits",
        )
        for field in fields:
            out.write(f"  const uint16_t *{field};\n")
        out.write("} tiny_bert_case_t;\n\n")
        out.write("extern const tiny_bert_case_t tiny_bert_cases[];\n")
        out.write("extern const int tiny_bert_case_count;\n\n")
        out.write("#endif\n")

    i32_fields = ("token_ids", "type_ids", "attention_mask")
    u16_fields = (
        "word_embeddings", "position_embeddings", "token_type_embeddings",
        "emb_ln_gamma", "emb_ln_beta", "attn_ln_gamma", "attn_ln_beta",
        "ffn_ln_gamma", "ffn_ln_beta", "wq", "wk", "wv", "wo", "w1", "w2",
        "bq", "bk", "bv", "bo", "b1", "b2", "pool_w", "pool_b",
        "classifier_w", "classifier_b", "expected_logits",
    )
    with source.open("w", encoding="utf-8") as out:
        out.write('#include "tiny_bert_cases.h"\n\n')
        for idx, (case, data) in enumerate(generated):
            prefix = f"tiny_bert_case_{idx}"
            out.write(f"/* {case.name}: layers={case.n_layers}, seq={case.seq_len}, d_model={case.d_model} */\n")
            for field in i32_fields:
                emit_i32_array(out, f"{prefix}_{field}", data[field])
            for field in u16_fields:
                emit_u16_array(out, f"{prefix}_{field}", data[field])

        out.write("const tiny_bert_case_t tiny_bert_cases[] = {\n")
        for idx, (case, data) in enumerate(generated):
            prefix = f"tiny_bert_case_{idx}"
            out.write("  {\n")
            out.write(f'    .name = "{case.name}",\n')
            out.write(f"    .seq_len = {case.seq_len},\n")
            out.write(f"    .vocab_size = {case.vocab_size},\n")
            out.write(f"    .type_vocab_size = {case.type_vocab_size},\n")
            out.write(f"    .d_model = {case.d_model},\n")
            out.write(f"    .n_heads = {case.n_heads},\n")
            out.write(f"    .head_dim = {case.head_dim},\n")
            out.write(f"    .hidden_dim = {case.hidden_dim},\n")
            out.write(f"    .n_layers = {case.n_layers},\n")
            out.write(f"    .num_classes = {case.num_classes},\n")
            out.write(f"    .scale_bf16 = 0x{data['scale_bf16']:04x},\n")
            out.write(f"    .tolerance_x100000 = {case.tolerance_x100000},\n")
            for field in i32_fields + u16_fields:
                out.write(f"    .{field} = {prefix}_{field},\n")
            out.write("  },\n")
        out.write("};\n\n")
        out.write("const int tiny_bert_case_count = sizeof(tiny_bert_cases) / sizeof(tiny_bert_cases[0]);\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "generated",
    )
    args = parser.parse_args()
    write_assets(DEFAULT_CASES, args.out_dir)
    print(f"wrote tiny-BERT cases to {args.out_dir} using PyTorch backend")


if __name__ == "__main__":
    main()
