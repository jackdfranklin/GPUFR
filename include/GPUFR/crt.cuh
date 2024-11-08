#pragma once

#include "GPUFR/types.h"
#include "GPUFR/ff_math.cuh"

void print_vec(const u32* vec, int size);
void print_vec(const u64* vec, int size);
void compute_crt(u32** in_arrs, u32* moduli, u64* out_arr, int arr_size, int n_moduli);
