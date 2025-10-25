#pragma once

#include "GPUFR/ff_math.cuh"
#include "GPUFR/cuda_safe_call.cuh"
#include "GPUFR/ntt.cuh"
#include "GPUFR/cuda_safe_call.cuh"

#include <vector>

__global__ void elementwise_multiply(u32* arr1, u32* arr2, u32 prime, int required_threads);

__global__ void fast_multiply(u32* &cu_in1, u32* &cu_in2, u32* &cu_output, int arr_size1, int arr_size2, std::vector<u32> ws, u32 prime);