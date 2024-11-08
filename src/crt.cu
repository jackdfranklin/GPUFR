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

    printf("m %llu \n", M);

    u64* Ms = new u64[n_moduli];
    u64* ys = new u64[n_moduli];

    for (int i=0; i<n_moduli; i++) 
    {
        Ms[i] = M/moduli[i];
        ys[i] = modular_inverse(Ms[i], static_cast<u64>(moduli[i]));
    }

    print_vec(ys, n_moduli);
    printf("\n");
    print_vec(Ms, n_moduli);
    printf("\n");
    print_vec(moduli, n_moduli);
    printf("\n");

    printf("ys: %llu Ms: %llu = %llu \n", ys[0], Ms[0], ff_multiply(ys[0], Ms[0], static_cast<u64>(moduli[0])));

    for (int i=0; i<arr_size; i++) 
    {
        out_arr[i] = 0;
        for (int j=0; j<n_moduli; j++) 
        {
            u64 partial_mult = mod_multiply(static_cast<u64>(in_arrs[j][i]), Ms[j], M);
            partial_mult = mod_multiply(partial_mult, ys[j], M);
            out_arr[i] = ff_add(out_arr[i], partial_mult, M);
            // out_arr[i] += Ms[j];
            printf("in_arrs: %llu Ms: %llu ys: %llu \n", in_arrs[j][i], Ms[j], ys[j]);
        }
        // out_arr[i] = out_arr[i]%M;
    }
}