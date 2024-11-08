#include "GPUFR/crt.cuh"
#include <stdio.h>

void print_vec(const u64* vec, int size)
{
    for (int i=0; i<size; i++)
    {
        printf("%i, ", vec[i]);
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
        M *= (u64)moduli[i];

    printf("m %u \n", M);

    u64* Ms = new u64[n_moduli];
    u64* ys = new u64[n_moduli];

    for (int i=0; i<n_moduli; i++) 
    {
        Ms[i] = M/moduli[i];
        ys[i] = modular_inverse(Ms[i], moduli[i]);
    }

    // print_vec(moduli, n_moduli);

    for (int i=0; i<arr_size; i++) 
    {
        out_arr[i] = 0;
        for (int j=0; j<n_moduli; j++) 
        {
            out_arr[i] += in_arrs[j][i] * Ms[j] * ys[j];
            // out_arr[i] += Ms[j];
        }
        out_arr[i] = out_arr[i]%M;
    }
}