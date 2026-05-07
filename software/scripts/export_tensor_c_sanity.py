#!/usr/bin/env python3
"""Export a tiny PyTorch BF16 tensor as linkable C assets for conversion sanity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_VALUES = (
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    -0.125,
    -0.0625,
    0.0,
    0.0625,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
)


def bf16_bits(tensor: torch.Tensor) -> list[int]:
    bf16 = tensor.detach().to(torch.bfloat16).contiguous().cpu()
    return [int(x) & 0xFFFF for x in bf16.view(torch.int16).reshape(-1).tolist()]


def bf16_bit_to_float(bits: int) -> float:
    signed_bits = bits if bits < 0x8000 else bits - 0x10000
    return (
        torch.tensor([signed_bits], dtype=torch.int16)
        .view(torch.bfloat16)
        .float()
        .item()
    )


def write_assets(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor = torch.tensor(DEFAULT_VALUES, dtype=torch.float32).reshape(4, 4)
    bits = bf16_bits(tensor)

    print("PYTORCH_TENSOR_C_SANITY")
    print("idx,row,col,float32,bf16_bits,bf16_float")
    flat = tensor.reshape(-1).tolist()
    for idx, (value, raw) in enumerate(zip(flat, bits)):
        print(f"{idx},{idx // 4},{idx % 4},{value:.8f},0x{raw:04x},{bf16_bit_to_float(raw):.8f}")

    header = out_dir / "tensor_c_sanity.h"
    source = out_dir / "tensor_c_sanity.c"

    with header.open("w", encoding="utf-8") as out:
        out.write("#ifndef TOYROCC_GENERATED_TENSOR_C_SANITY_H\n")
        out.write("#define TOYROCC_GENERATED_TENSOR_C_SANITY_H\n\n")
        out.write("#include <stdint.h>\n\n")
        out.write("#define TENSOR_C_SANITY_ROWS 4\n")
        out.write("#define TENSOR_C_SANITY_COLS 4\n")
        out.write("#define TENSOR_C_SANITY_SIZE 16\n\n")
        out.write("extern const uint16_t tensor_c_sanity_bf16[TENSOR_C_SANITY_SIZE];\n\n")
        out.write("#endif\n")

    with source.open("w", encoding="utf-8") as out:
        out.write('#include "tensor_c_sanity.h"\n\n')
        out.write("const uint16_t tensor_c_sanity_bf16[TENSOR_C_SANITY_SIZE] = {\n")
        for i in range(0, len(bits), 8):
            chunk = bits[i : i + 8]
            out.write("  " + ", ".join(f"0x{x:04x}" for x in chunk) + ",\n")
        out.write("};\n")

    print(f"wrote {header}")
    print(f"wrote {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "generated",
        help="Directory for generated C assets.",
    )
    args = parser.parse_args()
    write_assets(args.out_dir)


if __name__ == "__main__":
    main()
