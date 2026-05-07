#include "bfloat16_utils.h"
#include "generated/tensor_c_sanity.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
  printf("=== PyTorch Tensor -> C BF16 Sanity ===\n");
  printf("CSV_HEADER,idx,row,col,bf16_bits,c_float\n");

  for (int i = 0; i < TENSOR_C_SANITY_SIZE; i++) {
    const uint16_t raw = tensor_c_sanity_bf16[i];
    const float value = bf16_to_float(raw);
    printf("CSV_DATA,%d,%d,%d,0x%04x,", i, i / TENSOR_C_SANITY_COLS, i % TENSOR_C_SANITY_COLS, raw);
    print_float_inline(value);
    printf("\n");
  }

  return 0;
}

#include "generated/tensor_c_sanity.c"
