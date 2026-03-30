#include "rocc.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TEST_SIZE 8

static inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static inline float uq32_32_to_float(uint64_t fixed_val) {
    return (float)((double)fixed_val / 4294967296.0);
}

static inline uint64_t rocc_pass1(uint16_t *start, uint64_t length) {
    uint64_t raw_bits;
    asm volatile ("fence");
    ROCC_INSTRUCTION_DSS(0, raw_bits, start, length, 0);
    return raw_bits;
}

static inline uint64_t rocc_pass2(uint64_t *out) {
    uint64_t raw_bits;
    asm volatile ("fence");
    ROCC_INSTRUCTION_DS(0, raw_bits, out, 1);
    return raw_bits;
}

uint16_t input_bf16[TEST_SIZE] __attribute__((aligned(64)));
uint64_t output_uq32_32[TEST_SIZE] __attribute__((aligned(64)));

int main(void) {
    static const float input_vals[TEST_SIZE] = {
        0.0f, 0.5f, 1.0f, 1.5f,
        -0.5f, -1.0f, 2.0f, 0.25f,
    };

    uint64_t pass1_raw;
    uint64_t pass2_raw;
    float softmax_sum = 0.0f;

    printf("softmax_norm_fpga_smoke: starting\n");

    for (int i = 0; i < TEST_SIZE; i++) {
        input_bf16[i] = float_to_bf16(input_vals[i]);
        output_uq32_32[i] = 0;
    }

    pass1_raw = rocc_pass1(input_bf16, TEST_SIZE);
    pass2_raw = rocc_pass2(output_uq32_32);

    printf("pass1_raw=0x%016lx\n", (unsigned long)pass1_raw);
    printf("pass2_raw=0x%016lx\n", (unsigned long)pass2_raw);

    for (int i = 0; i < TEST_SIZE; i++) {
        float val = uq32_32_to_float(output_uq32_32[i]);
        softmax_sum += val;
        printf("out[%d]=0x%016lx %f\n", i, (unsigned long)output_uq32_32[i], val);
    }

    printf("softmax_sum=%f\n", softmax_sum);

    return 0;
}
