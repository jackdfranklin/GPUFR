#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip> 

#include "GPUFR/ff_math.cuh"
#include "GPUFR/cuda_safe_call.cuh"

std::vector<u32> get_w(const std::string& filename, int index);

void do_ntt(u32* &cu_array, u32* &cu_output, int arr_size, std::vector<u32> ws, u32 prime, bool inverse=false);
void do_bulk_ntt(u32* &cu_array, u32* &cu_output, int num_samps, int dimension, int mult_depth, std::vector<u32> ws, u32 prime, bool inverse=false);
