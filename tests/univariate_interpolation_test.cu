#include "GPUFR/vandermonde_solver.cuh"

#include <vector>
#include <algorithm>
#include <numeric>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <flint/flint.h>
#include <flint/nmod_types.h>
#include <flint/nmod.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

u32 evaluate_poly(u32 z, const std::vector<u32> &coefficients, nmod_t mod){
	u32 result = 0;
	for(size_t i = 0; i < coefficients.size(); i++){
		u32 term = nmod_mul(coefficients.at(i), nmod_pow_ui(z, i + 1, mod), mod);
		result = nmod_add(result, term, mod);
	}

	return result;
}

TEST_CASE("Univariate Polynomial Interpolation", "[Interpolation][Vandermonde]"){
	
	u32 p = GENERATE(2546604103, 3998191247);
	INFO("p = "<<p);
	nmod_t mod = {0};
	nmod_init(&mod, p);

	u32 number_of_values = 128;

	// Probe points need to be powers of the anchor point
	u32 anchor_point = GENERATE_COPY(take(1, random((u32)1, p-1)));
	std::vector<u32> v(number_of_values);
	std::iota(v.begin(), v.end(), 1);
	std::transform(v.begin(), v.end(), v.begin(), [anchor_point, mod](u32 exp){return nmod_pow_ui(anchor_point, exp, mod);});

	auto true_coeff = GENERATE_COPY(take(1, chunk(number_of_values, random((u32)1, p-1))));
	auto poly = [&true_coeff, mod](u32 z){

		return evaluate_poly(z, true_coeff, mod);
	};
	
	std::vector<u32> f(number_of_values);
	std::transform(v.begin(), v.end(), f.begin(), poly);

	std::stringstream f_string;
	for(auto fi: f){
		f_string<<fi<<" ";
	}

	INFO("f = "<<f_string.str());

	u32 *d_v;
	cudaMalloc(&d_v, number_of_values*sizeof(u32));
	cudaMemcpy(d_v, v.data(), number_of_values*sizeof(u32), cudaMemcpyHostToDevice);

	u32 *d_f;
	cudaMalloc(&d_f, number_of_values*sizeof(u32));
	cudaMemcpy(d_f, f.data(), number_of_values*sizeof(u32), cudaMemcpyHostToDevice);

	u32 *d_c;
	cudaMalloc(&d_c, number_of_values*sizeof(u32));
	cudaDeviceSynchronize();
	solve_transposed_vandermonde(number_of_values, d_v, d_f, p, d_c);
	cudaDeviceSynchronize();


	std::vector<u32> c(number_of_values);
	cudaMemcpy(c.data(), d_c, number_of_values*sizeof(u32), cudaMemcpyDeviceToHost);

	std::stringstream c_string;
	for(auto ci: c){
		c_string<<ci<<" ";
	}

	INFO("c = "<<c_string.str());

	SECTION("Polynomial should evaluate probes correctly"){
		for(size_t i = 0; i < number_of_values; i++){
			INFO("z["<<i<<"] = "<<v.at(i));
			REQUIRE(evaluate_poly(v.at(i), c, mod) == f.at(i)); 
		}
	}

}
