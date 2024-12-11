#pragma once

#include "GPUFR/types.hpp"
#include "GPUFR/ff_math.cuh"
#include <gmp.h>

void print_vec(const u32* vec, int size);

void print_vec(const u64* vec, int size);

void compute_crt(u32** in_arrs, u32* moduli, u64* out_arr, int arr_size, int n_moduli);

void compute_crt(u32** in_arrs, u32* moduli, mpz_t* out_arr, int arr_size, int n_moduli);

void compute_crt(mpz_t* out_arr, mpz_t* in1, mpz_t* in2, mpz_t mod_1, mpz_t mod_2, int arr_size);

void compute_crt(mpz_t* out_arr, mpz_t* in1, u32* in2, mpz_t mod_1, u32 mod_2, int arr_size);

