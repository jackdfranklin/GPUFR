#include "GPUFR/ntt.cuh"

#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

int as_int(u32 val, u32 prime)
{
    int result = val;
    if (result > prime/2) result = result - prime;
    return result;
}

void print_vec(const u32* vec, int size, u32 prime)
{
    for (int i=0; i<size; i++)
    {
        printf("%i, ", vec[i]);
    }
    printf("\n");
}

TEST_CASE("ntt_test"){
    int deviceCount = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    int arr_size = 4;

    u32* in_arr = new u32[arr_size];
    u32* out_arr = new u32[arr_size];

    for (int i=0; i<arr_size; i++)
    {
        in_arr[i] = i+1;
    }

    u32* d_in_arr;
    u32* d_out_arr;

    int bytes_arr = arr_size*sizeof(u32);
    CUDA_SAFE_CALL(cudaMalloc(&d_in_arr, bytes_arr));
    CUDA_SAFE_CALL(cudaMalloc(&d_out_arr, bytes_arr));

    CUDA_SAFE_CALL(cudaMemcpy(d_in_arr, in_arr, bytes_arr, cudaMemcpyHostToDevice));

    std::vector<u32> ws = get_w("./precomp/primes_roots_13.csv", 0);
    u32 prime = ws[0];
    do_ntt(d_in_arr, d_out_arr, arr_size, ws, prime);

    CUDA_SAFE_CALL(cudaMemcpy(out_arr, d_out_arr, bytes_arr, cudaMemcpyDeviceToHost));

    print_vec(out_arr, arr_size, prime);

    CUDA_SAFE_CALL(cudaFree(d_in_arr));
    CUDA_SAFE_CALL(cudaFree(d_out_arr));

    delete[] in_arr;
    delete[] out_arr;

    REQUIRE(true);
}

TEST_CASE("ntt_and_intt"){
    int deviceCount = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    int arr_size = 1<<8;

    u32* in_arr = new u32[arr_size];
    u32* out_arr = new u32[arr_size];

    for (int i=0; i<arr_size; i++)
    {
        in_arr[i] = i+1;
    }

    u32* d_in_arr;
    u32* d_out_arr;

    int bytes_arr = arr_size*sizeof(u32);
    CUDA_SAFE_CALL(cudaMalloc(&d_in_arr, bytes_arr));
    CUDA_SAFE_CALL(cudaMalloc(&d_out_arr, bytes_arr));

    CUDA_SAFE_CALL(cudaMemcpy(d_in_arr, in_arr, bytes_arr, cudaMemcpyHostToDevice));

    std::vector<u32> ws = get_w("./precomp/primes_roots_13.csv", 0);
    u32 prime = ws[0];
    do_ntt(d_in_arr, d_out_arr, arr_size, ws, prime);
    do_ntt(d_out_arr, d_in_arr, arr_size, ws, prime, true);

    CUDA_SAFE_CALL(cudaMemcpy(out_arr, d_in_arr, bytes_arr, cudaMemcpyDeviceToHost));

    for (int i=0; i<arr_size; i++)
    {
        REQUIRE(in_arr[i] == out_arr[i]);
    }

    CUDA_SAFE_CALL(cudaFree(d_in_arr));
    CUDA_SAFE_CALL(cudaFree(d_out_arr));

    delete[] in_arr;
    delete[] out_arr;
}

TEST_CASE("ntt_bulk"){
    int deviceCount = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    int exp = 2;
    int n_samps = (1<<exp) + 1;
    int initial_pol_size = 4;
    int arr_size = n_samps*(n_samps-1)*initial_pol_size;

    u32* in_arr = new u32[arr_size];
    u32* out_arr = new u32[arr_size];

    // for (int i=0; i<arr_size; i++)
    // {
    //     if (i%initial_pol_size == 0 || i%initial_pol_size == 1)
    //     {
    //         in_arr[i] = 0;
    //     } else if (i%initial_pol_size == 2)
    //     {
    //         in_arr[i] = 1;
    //     } else {
    //         in_arr[i] = i+1;
    //     }
    // }

    for (int i=0; i<arr_size; i++)
    {
        in_arr[i] = i%initial_pol_size + 1;
    }

    u32* d_in_arr;
    u32* d_out_arr;

    int bytes_arr = arr_size*sizeof(u32);
    CUDA_SAFE_CALL(cudaMalloc(&d_in_arr, bytes_arr));
    CUDA_SAFE_CALL(cudaMalloc(&d_out_arr, bytes_arr));

    CUDA_SAFE_CALL(cudaMemcpy(d_in_arr, in_arr, bytes_arr, cudaMemcpyHostToDevice));

    std::vector<u32> ws = get_w("./precomp/primes_roots_13.csv", 0);
    u32 prime = ws[0];
    do_bulk_ntt(d_in_arr, d_out_arr, n_samps, 0, ws, prime);
    do_bulk_ntt(d_out_arr, d_in_arr, n_samps, 0, ws, prime, true);

    CUDA_SAFE_CALL(cudaMemcpy(out_arr, d_in_arr, bytes_arr, cudaMemcpyDeviceToHost));

    // print_vec(out_arr, arr_size, prime);

    for (int i=0; i<arr_size; i++)
    {
        REQUIRE(in_arr[i] == out_arr[i]);
    }

    CUDA_SAFE_CALL(cudaFree(d_in_arr));
    CUDA_SAFE_CALL(cudaFree(d_out_arr));

    delete[] in_arr;
    delete[] out_arr;
}