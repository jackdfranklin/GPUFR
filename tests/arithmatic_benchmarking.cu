#include <stdio.h>
#include "GPUFR/types.hpp"
#include "GPUFR/cuda_safe_call.cuh"

#define N_SAMPS 1000000000
#define PRIME 1000112129
#define R 1<<31
#define RI 965821785
#define RUNS 10000

__global__ void ff_multiply_cast(u32* a, u32* b, u32* out, u32 p, int required_threads){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        u64 prod = (u64)(a[idx]%p) * (u64)(b[idx]%p);
        out[idx] = (u32)(prod%(u64)p);
    }
}

__global__ void ff_multiply_fancy(u32* a, u32* b, u32* out, u32 p, int required_threads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        u32 result = 0;
        a[idx] = a[idx] % p;  // Reduce a to avoid overflow in initial multiplication

        while (b[idx] > 0) {
            if (b[idx] % 2 == 1) {  // If b is odd, add a to result
                result = (result + a[idx]) % p;
            }
            a[idx] = (a[idx] * 2) % p;  // Double a
            b[idx] /= 2;           // Halve b
        }
        out[idx] = result;
    }
}



__global__ void ff_multiply_mont(u32* a, u32* b, u32* out, u32 p, int required_threads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        u32 mont_a = (a[idx]*R) % p;
        u32 mont_b = (b[idx]*R) % p;

        u32 T = mont_a*mont_b;

        u32 m = (T*(p-1))%R;

        u32 u = (T+m*p) / R;

        out[idx] = (u*RI) % p;
    }
}

int main()
{
    u32* as = new u32[N_SAMPS];
    u32* bs = new u32[N_SAMPS];
    u32* out1 = new u32[N_SAMPS];
    u32* out2 = new u32[N_SAMPS];

    std::srand(time(0));
    for (int i=0; i<N_SAMPS; i++)
    {
        as[i] = (std::rand())%PRIME;
        bs[i] = (std::rand())%PRIME;
    }

    u32 *d_as, *d_bs, *d_out1, *d_out2; 
    size_t bytes = N_SAMPS * sizeof(u32);

    CUDA_SAFE_CALL(cudaMalloc(&d_as, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_bs, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_out1, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_out2, bytes));

    CUDA_SAFE_CALL(cudaMemcpy(d_as, as, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_bs, bs, bytes, cudaMemcpyHostToDevice));

    int required_threads = N_SAMPS;
    int threadsPerBlock = required_threads>256? 256 : required_threads;
    int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;


    float time1, time2;
    float total_time1 = 0;
    float total_time2 = 0;
    float total_time3 = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ff_multiply_fancy<<<blocksPerGrid, threadsPerBlock>>>(d_as, d_bs, d_out1, PRIME, required_threads);


    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        ff_multiply_cast<<<blocksPerGrid, threadsPerBlock>>>(d_as, d_bs, d_out1, PRIME, required_threads);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time1 += milliseconds;
    }

    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        ff_multiply_mont<<<blocksPerGrid, threadsPerBlock>>>(d_as, d_bs, d_out2, PRIME, required_threads);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time2 += milliseconds;
    }

    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        ff_multiply_fancy<<<blocksPerGrid, threadsPerBlock>>>(d_as, d_bs, d_out2, PRIME, required_threads);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time3 += milliseconds;
    }

    std::printf("Average time with cast: %f \n", total_time1/RUNS);
    std::printf("Average time with mont: %f \n", total_time2/RUNS);
    std::printf("Average time with fancy: %f \n", total_time3/RUNS);

    // Checking results
    CUDA_SAFE_CALL(cudaMemcpy(out1, d_out1, bytes, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(out2, d_out2, bytes, cudaMemcpyDeviceToHost));

    for (int i=0; i<N_SAMPS; i++)
    {
        if (out1[i] != out2[i])
        {
            std::printf("Result missmach: %i != %i \n", out1, out2);
            return 1;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}