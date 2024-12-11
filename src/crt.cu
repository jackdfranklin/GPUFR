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

    mpz_clear(M);
    mpz_clear(p);
    mpz_clear(partial_mult);

    for (int i=0; i<n_moduli; i++) 
    {
        mpz_clear(Ms[i]);
        mpz_clear(ys[i]);
    }

    delete[] Ms;
    delete[] ys;
}

void compute_crt(mpz_t* out_arr, mpz_t* in1, mpz_t* in2, mpz_t mod_1, mpz_t mod_2, int arr_size)
{
    mpz_t M, p, partial_mult;
    mpz_init(M);
    mpz_init(p);
    mpz_init(partial_mult);
    mpz_set_ui(M, 1);

    mpz_t Ms[2];
    mpz_t ys[2];
    for (int i=0; i<2; i++) 
    {
        mpz_init(Ms[i]);
        mpz_init(ys[i]);
    }

    mpz_mul(M, M, mod_1);
    mpz_mul(M, M, mod_2);


    mpz_div(Ms[0], M, mod_1);
    mpz_invert(ys[0], Ms[0], mod_1);

    mpz_div(Ms[1], M, mod_1);
    mpz_invert(ys[1], Ms[1], mod_1);


    for (int i=0; i<arr_size; i++) 
    {
        mpz_mul(partial_mult, Ms[0], in1[i]);
        mpz_mod(partial_mult, partial_mult, M);
        mpz_mul(partial_mult, partial_mult,  ys[0]);
        mpz_mod(partial_mult, partial_mult, M);
        mpz_add(out_arr[i], out_arr[i], partial_mult);
        mpz_mod(out_arr[i], out_arr[i], M);

        mpz_mul(partial_mult, Ms[1], in2[i]);
        mpz_mod(partial_mult, partial_mult, M);
        mpz_mul(partial_mult, partial_mult,  ys[1]);
        mpz_mod(partial_mult, partial_mult, M);
        mpz_add(out_arr[i], out_arr[i], partial_mult);
        mpz_mod(out_arr[i], out_arr[i], M);
    }

    mpz_clear(M);
    mpz_clear(p);
    mpz_clear(partial_mult);

    for (int i=0; i<2; i++) 
    {
        mpz_clear(Ms[i]);
        mpz_clear(ys[i]);
    }
}

// todo make a function that will just append one moduli to the set