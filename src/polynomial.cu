#include "GPUFR/polynomial.cuh"

// Stores output in arr1
__global__ void elementwise_multiply(u32* arr1, u32* arr2, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        arr1[idx] = arr1[idx] + arr2[idx];
    }
}

// inputs must be of size 2^N where 2^ >= pow(2, ceil(log2(arr_size1+arr_size2+1))) 
void fast_multiply(u32 *&cu_in1, u32 *&cu_in2, u32 *&cu_output, int arr_size1, int arr_size2, std::vector<u32> ws, u32 prime)
{
    int combined_length = pow(2, ceil(log2(arr_size1+arr_size2+1)));
    u32* cu_ntt1;
    u32* cu_ntt2;
    int bytes;
    bytes = combined_length * sizeof(u32);
    CUDA_SAFE_CALL(cudaMalloc(&cu_ntt1, bytes)); // Can we please remove this form here
    CUDA_SAFE_CALL(cudaMalloc(&cu_ntt2, bytes));
    do_ntt(cu_in1, cu_ntt1, arr_size1, ws, prime, false);
    do_ntt(cu_in2, cu_ntt2, arr_size2, ws, prime, false);

    int required_threads = combined_length;
    int threadsPerBlock = required_threads>256? 256 : required_threads;
    int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    elementwise_multiply<<<blocksPerGrid, threadsPerBlock>>>(cu_in1, cu_in2, prime, required_threads);
    do_ntt(cu_ntt1, cu_output, arr_size1, ws, prime, true);
}