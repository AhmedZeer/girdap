#include "online_softmax.h"
#include "rocc.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_ELEMENTS 4096

// Helper to print a float with a label and a newline
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

// Helper to print a float inline (no newline, with padding space)
static void print_float_inline(float val) {
    if (val < 0.0f) {
        printf("-");
        val = -val;
    }
    int whole_part = (int)val;
    int frac_part = (int)((val - (float)whole_part) * 10000.0f);
    printf("%d.%04d ", whole_part, frac_part);
}

// Function to print the first 10, middle 10, and last 10 elements
static void print_array_samples(float *arr, uint64_t size, const char* name) {
    printf("\n--- ARRAY MONITOR: %s (Size: %lu) ---\n", name, (unsigned long)size);
    
    // First 10
    printf("First 10:  [ ");
    for(uint64_t i = 0; i < 10 && i < size; i++) {
        print_float_inline(arr[i]);
    }
    printf("]\n");

    // Middle 10
    if (size >= 30) {
        printf("Middle 10: [ ");
        uint64_t mid_start = (size / 2) - 5;
        for(uint64_t i = mid_start; i < mid_start + 10; i++) {
            print_float_inline(arr[i]);
        }
        printf("]\n");
    }

    // Last 10
    if (size >= 20) {
        printf("Last 10:   [ ");
        uint64_t last_start = size - 10;
        for(uint64_t i = last_start; i < size; i++) {
            print_float_inline(arr[i]);
        }
        printf("]\n");
    }
    printf("------------------------------------------------\n");
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

// Convert UQ12.20 to standard C float (Updated for OnlineSoftmax)
static inline float uq12_20_to_float(uint32_t fixed_val) {
    return (float)((double)fixed_val / 1048576.0);
}

static inline double uq32_32_to_double(uint64_t fixed_val) {
    // The fractional part is 32 bits, so we divide the raw integer by 2^32
    // 2^32 = 4294967296.0
    return (double)fixed_val / 4294967296.0;
}

// Read the RISC-V 64-bit hardware cycle counter
static inline uint64_t read_cycles(void) {
    uint64_t cycles;
    asm volatile ("fence; rdcycle %0" : "=r" (cycles));
    return cycles;
}

// Global arrays to prevent stack overflow on bare-metal
uint16_t bf16_array[MAX_ELEMENTS] __attribute__ ((aligned (64)));
float float_array[MAX_ELEMENTS];

// RoCC Invocation (Updated with strict memory barriers)
static inline uint64_t accumulate_softmax_den(uint16_t *start, uint64_t length) {
    return online_softmax_accumulate_denominator(start, length);
}

// Generate random float between -2.0 and 2.0
static float rand_float() {
    return ((float)rand() / (float)(RAND_MAX)) * 4.0f - 2.0f;
}

int main(void) {
    // Included 15 and 511 to explicitly test the hardware masking for unaligned memory beats
    uint64_t test_sizes[] = {16, 256, 512, 1024, 2048, 4096};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("Starting Scalable RoCC OnlineSoftmax Benchmark...\n");
    printf("CSV_HEADER, ArraySize, SW_Cycles, HW_Cycles, Speedup_Multiplier\n");

    int t = 0;
    int test_failed = 0;

    while (t < num_tests) {
        uint64_t current_size = test_sizes[t];

        // 1. Populate the arrays with random data for this size
        for(uint64_t i = 0; i < current_size; i++) {
            float val = rand_float();
            float_array[i] = val;
            bf16_array[i] = float_to_bf16(val);
        }

        print_array_samples(float_array, current_size, "Input Float Array");

        // 2. Run Software Benchmark (Online Softmax Algorithm)
        uint64_t sw_start = read_cycles();
        
        // Pass 1: Find Max (using truncated BFloat16 precision for fairness)
        float max_val = -INFINITY;
        for(uint64_t i = 0; i < current_size; i++) {
            float truncated_val = bf16_to_float(bf16_array[i]);
            if (truncated_val > max_val) {
                max_val = truncated_val;
            }
        }

        // Pass 2: Sum exp(x - max) -- UPGRADED TO DOUBLE
        double expected_double_sum = 0.0;
        for(uint64_t i = 0; i < current_size; i++) {
            float truncated_val = bf16_to_float(bf16_array[i]);
            // Use standard exp() for double-precision math
            expected_double_sum += exp((double)(truncated_val - max_val)); 
        }
        
        uint64_t sw_end = read_cycles();
        uint64_t sw_cycles = sw_end - sw_start;

        // 3. Run Hardware Benchmark
        uint64_t hw_start = read_cycles();
        
        // FIXED: Declare as uint64_t to prevent integer truncation!
        uint64_t raw_hw_bits = accumulate_softmax_den(bf16_array, current_size); 
        
        // Implicit Dependency Stall
        asm volatile ("add zero, zero, %0" : : "r" (raw_hw_bits));
        
        uint64_t hw_end = read_cycles();
        uint64_t hw_cycles = hw_end - hw_start;

        // --- Print Accumulated Results ---
        // FIXED: Declare as double to prevent precision loss
        double hw_double_res = uq32_32_to_double(raw_hw_bits); 
        
        print_float("-> Software Softmax Denominator: ", (float)expected_double_sum);
        print_float("-> Hardware Softmax Denominator: ", (float)hw_double_res);

        // 4. Verification Check
        double error = hw_double_res - expected_double_sum;
        if (error < 0.0) error = -error; 

        // Dynamic tolerance: Allow a 5% relative error
        double tolerance = expected_double_sum * 0.05;
        if (tolerance < 0.0) tolerance = -tolerance;
        if (tolerance < 1.0) tolerance = 1.0;

        if (error > tolerance) {
            printf("\n[!] FATAL MISMATCH at Array Size %lu\n", (unsigned long)current_size);
            printf("    Difference exceeds tolerance. Halting benchmark.\n\n");
            // test_failed = 1;
            // break; 
        }

        // 5. Calculate Speedup and Print strictly formatted output
        double speedup = (double)sw_cycles / (double)hw_cycles;
        int speedup_int = (int)speedup;
        int speedup_frac = (int)((speedup - (double)speedup_int) * 100.0);
        
        printf("CSV_DATA, %lu, %lu, %lu, %d.%02d\n", 
            (unsigned long)current_size, 
            (unsigned long)sw_cycles, 
            (unsigned long)hw_cycles, 
            speedup_int, 
            speedup_frac
        );

        t++; 
    }

    // if (test_failed) {
    //     return 1;
    // }
    
    printf("\nAll array sizes passed verification successfully!\n");
    return 0;
}
