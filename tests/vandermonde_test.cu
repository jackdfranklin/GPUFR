#include "GPUFR/vandermonde_solver.cuh"

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

std::vector<u32> master_poly_coefficients_function(const std::vector<u32> &v, nmod_t mod){
	
	std::vector<u32> c(v.size());

	c[v.size()-1] = nmod_sub(0, v.at(0), mod);

	for(size_t i = 1; i < v.size(); i++){
		for(size_t j = v.size() - 1 - i; j < v.size() - 1; j++){
			c[j] = nmod_sub(c[j], nmod_mul(v.at(i), c.at(j + 1), mod), mod);
		}

		c[v.size() - 1] = nmod_sub(c[v.size() - 1], v.at(i), mod);

	}

	return c;
}

void master_poly_coefficients_iterate(u32 i, const std::vector<u32> &v, std::vector<u32> &c, nmod_t mod){

	for(size_t j = v.size() - 1 - i; j < v.size() - 1; j++){
		c[j] = nmod_sub(c[j], nmod_mul(v.at(i), c.at(j + 1), mod), mod);
	}

	c[v.size() - 1] = nmod_sub(c[v.size() - 1], v.at(i), mod);
}

TEST_CASE("Solving Vandermonde Systems", "[Interpolation]"){
	
	std::random_device rand_device;
	std::mt19937 mersenne_engine {rand_device()};
	std::uniform_int_distribution<u32> dist {0, UINT32_MAX};

	u32 p = 2546604103;

	auto generator = [&](){
		return dist(mersenne_engine)%p;
	};

	u32 number_of_values = 4;
	std::vector<u32> v(number_of_values);

	std::generate(v.begin(), v.end(), generator);

	u32 *d_v, *d_c_in, *d_c;

	cudaMalloc(&d_v, number_of_values*sizeof(u32));
	cudaMalloc(&d_c_in, number_of_values*sizeof(u32));
	cudaMalloc(&d_c, number_of_values*sizeof(u32));

	cudaMemcpy(d_v, v.data(), number_of_values*sizeof(u32), cudaMemcpyHostToDevice);


	nmod_t mod = {0};
	nmod_init(&mod, p);

	std::vector<u32> c_reference(number_of_values);
	c_reference[number_of_values - 1] = nmod_sub(0, v.at(0), mod);

	cudaMemcpy(d_c_in, c_reference.data(), number_of_values*sizeof(u32), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	//INFO("Copying Coefficients From Device To Host...");
	std::vector<u32> c(number_of_values);

	for(size_t i = 1; i < number_of_values; i++){
		master_poly_coefficients_iterate(i, v, c_reference, mod);

		master_poly_coefficient_iter<<<1, number_of_values>>>(number_of_values, i, d_v, d_c_in, p, d_c);
		cudaMemcpy(c.data(), d_c, number_of_values*sizeof(u32), cudaMemcpyDeviceToHost);
		for(size_t j = 0; j < number_of_values; j++){
			INFO("i = "<<i);
			INFO("j = "<<j);
			REQUIRE(c.at(j) == c_reference.at(j));
		}

		u32 *temp = d_c_in;
		d_c_in = d_c;
		d_c = temp;
	}

	cudaFree(d_v);
	cudaFree(d_c_in);
	cudaFree(d_c);

}
