#ifndef TOYROCC_BFLOAT16_UTILS_H
#define TOYROCC_BFLOAT16_UTILS_H

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static inline uint16_t float_to_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return (uint16_t)(bits >> 16);
}

static inline float bf16_to_float(uint16_t value) {
  uint32_t bits = ((uint32_t)value) << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

static inline float fixed_s32_to_float(int32_t value, int frac_bits) {
  return (float)((double)value / (double)(1ULL << frac_bits));
}

static inline float fixed_s64_to_float(int64_t value, int frac_bits) {
  const double scale = (double)(1ULL << frac_bits);
  return (float)((double)value / scale);
}

static inline int16_t bf16_to_fixed_s16_sat(uint16_t value, int frac_bits) {
  const uint32_t sign = (uint32_t)(value >> 15);
  const uint32_t exp = (uint32_t)((value >> 7) & 0xFFu);
  const uint32_t mantissa = (uint32_t)(value & 0x7Fu);

  if ((value & 0x7FFFu) == 0) {
    return 0;
  }

  const uint32_t ext_mantissa = (exp == 0u) ? mantissa : (0x80u | mantissa);
  const int shift_amt = (((exp == 0u) ? 1 : (int)exp) - 127) - 7 + frac_bits;

  uint64_t shifted_mag;
  if (shift_amt >= 0) {
    shifted_mag = (shift_amt >= 56) ? UINT64_MAX : ((uint64_t)ext_mantissa << shift_amt);
  } else {
    const int right_shift = -shift_amt;
    shifted_mag = (right_shift >= 64) ? 0u : ((uint64_t)ext_mantissa >> right_shift);
  }

  if (sign == 0u) {
    if (shifted_mag > (uint64_t)INT16_MAX) {
      return INT16_MAX;
    }
    return (int16_t)shifted_mag;
  }

  if (shifted_mag >= 0x8000u) {
    return INT16_MIN;
  }
  return (int16_t)(-(int32_t)shifted_mag);
}

static inline uint16_t fixed_s64_to_bf16_sat(int64_t value, int frac_bits) {
  if (value == 0) {
    return 0;
  }

  const uint64_t raw_bits = (uint64_t)value;
  const uint16_t sign = (uint16_t)(raw_bits >> 63);
  const uint64_t mag = sign ? ((~raw_bits) + 1ULL) : raw_bits;

#if defined(__GNUC__) || defined(__clang__)
  const int msb_idx = 63 - __builtin_clzll(mag);
#else
  int msb_idx = 0;
  uint64_t tmp = mag;
  while (tmp >>= 1) {
    msb_idx++;
  }
#endif

  const int unbiased_exp = msb_idx - frac_bits;
  if (unbiased_exp <= -127) {
    return 0;
  }
  if (unbiased_exp > 127) {
    return (uint16_t)((sign << 15) | 0x7F7Fu);
  }

  uint64_t normalized_sig;
  if (msb_idx >= 7) {
    normalized_sig = mag >> (msb_idx - 7);
  } else {
    normalized_sig = mag << (7 - msb_idx);
  }

  const uint16_t exponent = (uint16_t)(unbiased_exp + 127);
  const uint16_t mantissa = (uint16_t)(normalized_sig & 0x7Fu);
  return (uint16_t)((sign << 15) | (exponent << 7) | mantissa);
}

static inline void print_float(const char *label, float value) {
  if (value < 0.0f) {
    printf("%s-", label);
    value = -value;
  } else {
    printf("%s", label);
  }

  const int whole_part = (int)value;
  const int frac_part = (int)((value - (float)whole_part) * 10000.0f);
  printf("%d.%04d\n", whole_part, frac_part);
}

static inline void print_float_inline(float value) {
  if (value < 0.0f) {
    printf("-");
    value = -value;
  }

  const int whole_part = (int)value;
  const int frac_part = (int)((value - (float)whole_part) * 10000.0f);
  printf("%d.%04d ", whole_part, frac_part);
}

#endif
