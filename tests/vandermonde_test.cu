#include "GPUFR/vandermonde_solver.cuh"

#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <flint/flint.h>
#include <flint/nmod_types.h>
#include <flint/nmod.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

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

u32 evaluate_master_polynomial(u32 z, const std::vector<u32> &c, nmod_t mod){
	u32 result = c.at(0);
	for(size_t i = 1; i < c.size(); i++){
		result = nmod_add(result, nmod_mul(c.at(i), nmod_pow_ui(z, i, mod), mod), mod); 
	}

	result = nmod_add(result, nmod_pow_ui(z, c.size(), mod), mod); //Coefficient for largest power is always 1

	return result;
}

u32 product_expansion(u32 z, const std::vector<u32> &v, nmod_t mod){
	u32 result = 1;
	for(size_t i = 0; i < v.size(); i++){
		result = nmod_mul(result, nmod_sub(z, v.at(i), mod), mod);
	}

	return result;
}

TEST_CASE("Solving Vandermonde Systems", "[Interpolation][Vandermonde]"){
	
	u32 p = GENERATE(13, 2546604103, 3998191247);
	INFO("p = "<<p);
	u32 number_of_values = GENERATE(64, 128);
	INFO("n = "<<number_of_values);

	auto v = GENERATE_COPY(take(5, chunk(number_of_values, random((u32)0, p-1))));

	std::stringstream v_string;
	for(auto vi: v){
		v_string<<vi<<" ";
	}

	INFO("v = "<<v_string.str());

	u32 *d_v, *d_c;

	cudaMalloc(&d_v, number_of_values*sizeof(u32));
	cudaMalloc(&d_c, number_of_values*sizeof(u32));

	cudaMemcpy(d_v, v.data(), number_of_values*sizeof(u32), cudaMemcpyHostToDevice);

	nmod_t mod = {0};
	nmod_init(&mod, p);
	std::vector<u32> c_reference = master_poly_coefficients_function(v, mod);

	std::vector<u32> c(number_of_values);
	extract_master_poly_coefficients(number_of_values, d_v, p, d_c);
	cudaMemcpy(c.data(), d_c, number_of_values*sizeof(u32), cudaMemcpyDeviceToHost);

	std::stringstream c_string;
	for(auto ci: c){
		c_string<<ci<<" ";
	}

	INFO("c = "<<c_string.str());

	SECTION("Same coefficients calculated on GPU and CPU"){
		for(size_t j = 0; j < number_of_values; j++){
			INFO("j = "<<j);
			REQUIRE(c.at(j) == c_reference.at(j));
		}
	}

	SECTION("Evaluation of master polynomial is correct"){
		auto z = GENERATE_COPY(take(10, random((u32)1, p-1)));
		INFO("z = "<<z);
		REQUIRE(product_expansion(z, v, mod) == evaluate_master_polynomial(z, c, mod));	
	}

	cudaFree(d_v);
	cudaFree(d_c);

}
