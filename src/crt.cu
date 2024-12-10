#include "GPUFR/crt.cuh"
#include <stdio.h>

void print_vec(const u64* vec, int size)
{
    for (int i=0; i<size; i++)
    {
        printf("%llu, ", vec[i]);
    }
}

void print_vec(const u32* vec, int size)
{
    for (int i=0; i<size; i++)
    {
        printf("%i, ", vec[i]);
    }
}

void compute_crt(u32** in_arrs, u32* moduli, u64* out_arr, int arr_size, int n_moduli)
{
    u64 M = 1;
    for (int i=0; i<n_moduli; i++) 
        M *= static_cast<u64>(moduli[i]);

    u64* Ms = new u64[n_moduli];
    u64* ys = new u64[n_moduli];

    for (int i=0; i<n_moduli; i++) 
    {
        Ms[i] = M/moduli[i];
        ys[i] = modular_inverse(Ms[i], static_cast<u64>(moduli[i]));
    }

    for (int i=0; i<arr_size; i++) 
    {
        out_arr[i] = 0;
        for (int j=0; j<n_moduli; j++) 
        {
            u64 partial_mult = mod_multiply(static_cast<u64>(in_arrs[j][i]), Ms[j], M);
            partial_mult = mod_multiply(partial_mult, ys[j], M);
            out_arr[i] = ff_add(out_arr[i], partial_mult, M);
        }
    }
}

void compute_crt(u32** in_arrs, u32* moduli, mpz_t* out_arr, int arr_size, int n_moduli)
{
    mpz_t M, p, partial_mult;
    mpz_init(M);
    mpz_init(p);
    mpz_init(partial_mult);
    mpz_set_ui(M, 1);

    mpz_t* Ms = new mpz_t[n_moduli];
    mpz_t* ys = new mpz_t[n_moduli];
    for (int i=0; i<n_moduli; i++) 
    {
        mpz_init(Ms[i]);
        mpz_init(ys[i]);

        mpz_mul_ui(M, M, moduli[i]);
    }

    for (int i=0; i<n_moduli; i++) 
    {
        mpz_set_ui(p, moduli[i]);
        mpz_div_ui(Ms[i], M, moduli[i]);
        mpz_invert(ys[i], Ms[i], p);
    }

    for (int i=0; i<arr_size; i++) 
    {
        mpz_set_ui(out_arr[i], 0);
        for (int j=0; j<n_moduli; j++) 
        {
            mpz_mul_ui(partial_mult, Ms[j], in_arrs[j][i]);
            mpz_mod(partial_mult, partial_mult, M);
            mpz_mul(partial_mult, partial_mult,  ys[j]);
            mpz_mod(partial_mult, partial_mult, M);
            mpz_add(out_arr[i], out_arr[i], partial_mult);
            mpz_mod(out_arr[i], out_arr[i], M);
        }
    }
}