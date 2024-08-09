#include "GPUFR/ff_math.cuh"
#include "GPUFR/types.h"

#include <array>
#include <vector>
#include <algorithm>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

#include <flint/flint.h>
#include <flint/nmod_types.h>
#include <flint/nmod.h>

#include <catch2/catch_test_macros.hpp>

__global__ void ff_add_test(u32 p, u32 *a, u32 *b, u32 *c){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	c[idx] = ff_add(a[idx], b[idx], p);	
}

TEST_CASE("First Test", "[Finite Field]"){
	// The setup is repeated before running each section
	std::array<u32, 3> p_array = {13, 2546604103, 3998191247};

	u32 number_of_values = 128;
	std::vector<u32> a(number_of_values), b(number_of_values);

	std::random_device rand_device;
	std::mt19937 mersenne_engine {rand_device()};
	std::uniform_int_distribution<u32> dist {0, ULONG_MAX};

	auto generator = [&](){
		return dist(mersenne_engine);
	};

	std::generate(a.begin(), a.end(), generator);
	std::generate(b.begin(), b.end(), generator);

	u32 *d_a, *d_b, *d_c;

	cudaMalloc(&d_a, number_of_values*sizeof(u32));
	cudaMalloc(&d_b, number_of_values*sizeof(u32));
	cudaMalloc(&d_c, number_of_values*sizeof(u32));

	cudaMemcpy(d_a, a.data(), number_of_values*sizeof(u32), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), number_of_values*sizeof(u32), cudaMemcpyHostToDevice);

	SECTION("Addition"){

		for(u32 p: p_array){
			
			ff_add_test<<<1,128>>>(p, d_a, d_b, d_c);
			cudaDeviceSynchronize();
			std::vector<u32> c(number_of_values);
			cudaMemcpy(c.data(), d_c, number_of_values*sizeof(u32), cudaMemcpyDeviceToHost);

			nmod_t modulus = {0};
			nmod_init(&modulus, p);

			for(size_t i = 0; i < number_of_values; i++){
				INFO("a = "<<a.at(i)<<", b = "<<b.at(i)<<", c = "<<c[i]);
				REQUIRE(c[i] == nmod_add(a.at(i)%p, b.at(i)%p, modulus));
			}

		}
		
	}
}
