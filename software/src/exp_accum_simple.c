#include "online_softmax.h"
#include "rocc.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// Helper to print floats using only integer printf
static void print_float(const char* label, float val) {
    if (val < 0.0f) {
        printf("%s-", label);
        val = -val;
    } else {
        printf("%s", label);
    }

    int whole_part = (int)val;
    int frac_part = (int)((val - (float)whole_part) * 10000.0f);
    
    printf("%d.%04d\n", whole_part, frac_part);
}

// Helper to cast a 32-bit float to a 16-bit bfloat16
static inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

// Helper to cast a 16-bit bfloat16 back to a 32-bit float
static inline float bf16_to_float(uint16_t bfloat) {
    uint32_t bits = ((uint32_t)bfloat) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// --- UPDATED: Convert UQ32.32 to standard C float ---
static inline float uq32_32_to_float(uint64_t fixed_val) {
    // 2^32 = 4294967296.0
    return (float)((double)fixed_val / 4294967296.0);
}

// Read the RISC-V 64-bit hardware cycle counter
static inline uint64_t read_cycles(void) {
    uint64_t cycles;
    asm volatile ("fence; rdcycle %0" : "=r" (cycles));
    return cycles;
}

#ifndef TEST_ARRAY_SIZE
#define TEST_ARRAY_SIZE 32
#endif

uint16_t bf16_array[TEST_ARRAY_SIZE] __attribute__ ((aligned (64)));

static inline uint64_t accumulate_softmax_den(uint16_t *start, uint64_t length) {
    return online_softmax_accumulate_denominator(start, length);
}

int main(void) {
    static const float seed_vals[] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        -4.0f, -1.5f, -2.5f, 0.5f,

        0.0f, 1.1f, 0.0f, 0.0f,
        0.0f, 0.0f, 3.0f , 0.0f,

        0.0f, 1.0f, 2.0f, 3.0f,
        -4.0f, -1.5f, -2.5f, 0.5f,

        0.0f, 1.1f, 0.0f, 0.0f,
        0.0f, 0.0f, 3.0f , 0.0f,

        0.0f, 1.0f, 2.0f, 3.0f,
        -4.0f, -1.5f, -2.5f, 0.5f,
    };
    const int seed_count = (int)(sizeof(seed_vals) / sizeof(seed_vals[0]));
    
    printf("Populating BFloat16 Array...\n");
    for(int i = 0; i < TEST_ARRAY_SIZE; i++) {
        float v = (i < seed_count) ? seed_vals[i] : 0.0f;
        bf16_array[i] = float_to_bf16(v);
    }

    // --- SOFTWARE BENCHMARK (Updated for Online Softmax Denominator) ---
    printf("\nExecuting Software (libm expf with max subtraction)...\n");
    uint64_t sw_start = read_cycles();
    
    // 1. Find the maximum (using truncated BFloat16 precision for fairness)
    float max_val = -INFINITY;
    for(int i = 0; i < TEST_ARRAY_SIZE; i++) {
        float truncated_val = bf16_to_float(bf16_array[i]);
        if (truncated_val > max_val) {
            max_val = truncated_val;
        }
    }

    // 2. Compute sum of exp(x_i - max)
    float expected_float_sum = 0.0f;
    for(int i = 0; i < TEST_ARRAY_SIZE; i++) {
        float truncated_val = bf16_to_float(bf16_array[i]);
        expected_float_sum += expf(truncated_val - max_val);
    }
    
    uint64_t sw_end = read_cycles();
    uint64_t sw_cycles = sw_end - sw_start;

    // Simulate what the expected float should look like in UQ32.32 format
    uint64_t expected_uq32_32 = (uint64_t)((double)expected_float_sum * 4294967296.0);

    // --- HARDWARE BENCHMARK ---
    printf("Executing RoCC Softmax Denominator Accumulator...\n");
    uint64_t hw_start = read_cycles();
    
    uint64_t raw_hw_bits = accumulate_softmax_den(bf16_array, TEST_ARRAY_SIZE);
    
    // Implicit Dependency Stall
    asm volatile ("add zero, zero, %0" : : "r" (raw_hw_bits));
    
    uint64_t hw_end = read_cycles();
    uint64_t hw_cycles = hw_end - hw_start;

    // --- RESULTS PARSING (Updated for UQ12.20) ---
    float hw_float_res = uq32_32_to_float(raw_hw_bits);

    // Extract 32-bit Integer and 32-bit Fraction for display
    uint32_t hw_int_part  = (uint32_t)(raw_hw_bits >> 32);
    uint32_t hw_frac_part = (uint32_t)(raw_hw_bits & 0xFFFFFFFFu);
    uint32_t sw_int_part  = (uint32_t)(expected_uq32_32 >> 32);
    uint32_t sw_frac_part = (uint32_t)(expected_uq32_32 & 0xFFFFFFFFu);

    // --- DECIMAL RESULTS ---
    printf("\n--- RESULTS (DECIMAL) ---\n");
    print_float("Software Expected Float: ", expected_float_sum);
    print_float("Hardware Computed Float: ", hw_float_res);
    
    // --- BIT-BY-BIT UQ12.20 COMPARISON (HEX) ---
    printf("\n--- UQ32.32 BIT COMPARISON (HEX) ---\n");
    printf("Format: [Integer 32b] . [Fraction 32b]\n");
    
    // %08x prints 32 bits (8 hex chars)
    printf("Expected UQ32.32: 0x%08x . 0x%08x\n", sw_int_part, sw_frac_part);
    printf("Hardware UQ32.32: 0x%08x . 0x%08x\n", hw_int_part, hw_frac_part);
    
    printf("XOR Difference:   0x%08x . 0x%08x\n", (sw_int_part ^ hw_int_part), (sw_frac_part ^ hw_frac_part));

    // --- PERFORMANCE ---
    printf("\n--- PERFORMANCE ---\n");
    printf("Software Cycles: %lu\n", (unsigned long)sw_cycles);
    printf("Hardware Cycles: %lu\n", (unsigned long)hw_cycles);
    
    if (hw_cycles < sw_cycles) {
        unsigned long speedup = (unsigned long)(sw_cycles / hw_cycles);
        printf("Hardware is FASTER! Approx Speedup: %lux\n", speedup);
    } else {
        printf("Hardware is SLOWER by %lu cycles.\n", (unsigned long)(hw_cycles - sw_cycles));
    }

    // --- TOLERANCE CHECK ---
    float error = hw_float_res - expected_float_sum;
    if (error < 0) error = -error; 
    
    // BFloat16 polynomial approximation tolerance
    if (error < 0.05f) { // Tightened tolerance since exp(x-max) outputs are bounded 0.0 to 1.0
        printf("\nSuccess! Hardware UQ12.20 matches expected mathematical model.\n");
        return 0;
    }
    
    printf("\nFailure: Hardware UQ12.20 diverges too much from expected float.\n");
    return 0;
}
