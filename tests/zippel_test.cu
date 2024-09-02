#include "GPUFR/zippel.cuh"

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

TEST_CASE("Multivariate polynomial interpolation using Zippel's algorithm"){

	u32 p = GENERATE(2546604103, 3998191247);
	nmod_t mod = {0};
	nmod_init(&mod, p);

	SECTION("Monomial evaluation"){
		u32 n_vars= GENERATE(take(1, random((u32)1, (u32)10)));

		std::vector<u32> exponents = GENERATE_COPY(take(1, chunk(128 * n_vars, random((u32)1, (u32)20))));
		u32 *d_e;
		cudaMalloc(&d_e, n_vars*128*sizeof(u32));
		cudaMemcpy(d_e, exponents.data(), n_vars*128*sizeof(u32), cudaMemcpyHostToDevice);

		std::vector<u32> anchor_points = GENERATE_COPY(take(1, chunk(n_vars, random((u32)2, p-1))));
		u32 *d_a;
		cudaMalloc(&d_a, n_vars*sizeof(u32));
		cudaMemcpy(d_a, anchor_points.data(), n_vars*sizeof(u32), cudaMemcpyHostToDevice);
		
		std::vector<u32> reference(128);
		for(size_t i = 0; i < 128; i++){
			u32 r = 1;
			for(size_t j = 0; j < n_vars; j++){
				r = nmod_mul(r, nmod_pow_ui(anchor_points.at(j), exponents.at(i + 128*j), mod), mod);
			}
			reference.at(i) = r;
		}

		u32 *d_r;
		cudaMalloc(&d_r, 128*sizeof(u32));
		evaluate_monomials<<<1,128>>>(n_vars, 128, d_a, d_e, d_r, p);

		std::vector<u32> result(128);
		cudaMemcpy(result.data(), d_r, 128*sizeof(u32), cudaMemcpyDeviceToHost);

		for(size_t i = 0; i < 128; i++){
			REQUIRE(result.at(i) == reference.at(i));
		}
		
	}

	SECTION("Polynomial 1"){
		auto poly = "3 + x*y + x^2 + 4*y^3";

		//auto reconst_poly = zippel_reconstruct(poly, 2, 16);

	}
}
