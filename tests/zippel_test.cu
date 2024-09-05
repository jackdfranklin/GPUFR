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
	INFO("p = "+std::to_string(p));
	nmod_t mod = {0};
	nmod_init(&mod, p);

	SECTION("Monomial evaluation"){
		u32 n_vars= GENERATE(take(4, random((u32)1, (u32)10)));
		size_t n_values = GENERATE(take(4, random((u32)32, (u32)128)));

		std::vector<u32> exponents = GENERATE_COPY(take(1, chunk(n_values * n_vars, random((u32)1, (u32)20))));
		u32 *d_e;
		size_t pitch = 0;
		cudaMallocPitch(&d_e, &pitch, n_values*sizeof(u32), n_vars);
		cudaMemcpy2D(d_e, pitch, exponents.data(), n_values*sizeof(u32), n_values*sizeof(u32), n_vars, cudaMemcpyHostToDevice);
		//cudaMalloc(&d_e, n_vars*n_values*sizeof(u32));
		//cudaMemcpy(d_e, exponents.data(), n_vars*n_values*sizeof(u32), cudaMemcpyHostToDevice);

		std::vector<u32> anchor_points = GENERATE_COPY(take(1, chunk(n_vars, random((u32)2, p-1))));
		u32 *d_a;
		cudaMalloc(&d_a, n_vars*sizeof(u32));
		cudaMemcpy(d_a, anchor_points.data(), n_vars*sizeof(u32), cudaMemcpyHostToDevice);
		
		std::vector<u32> reference(n_values);
		for(size_t i = 0; i < n_values; i++){
			u32 r = 1;
			for(size_t j = 0; j < n_vars; j++){
				r = nmod_mul(r, nmod_pow_ui(anchor_points.at(j), exponents.at(i + n_values*j), mod), mod);
			}
			reference.at(i) = r;
		}

		u32 *d_r;
		cudaMalloc(&d_r, n_values*sizeof(u32));
		evaluate_monomials<<<1,n_values>>>(n_vars, pitch/sizeof(u32), d_a, d_e, d_r, p);

		std::vector<u32> result(n_values);
		cudaMemcpy(result.data(), d_r, n_values*sizeof(u32), cudaMemcpyDeviceToHost);

		for(size_t i = 0; i < n_values; i++){
			REQUIRE(result.at(i) == reference.at(i));
		}

		cudaFree(d_e);
		cudaFree(d_a);
		cudaFree(d_r);
		
	}

	SECTION("Power evaluation"){

		u32 n_vars= GENERATE(take(4, random((u32)1, (u32)10)));
		size_t n_values = GENERATE(take(4, random((u32)8, (u32)16)));
		
		std::vector<u32> anchor_points = GENERATE_COPY(take(1, chunk(n_vars, random((u32)2, p-1))));

		u32 *d_a;
		cudaMalloc(&d_a, n_vars*sizeof(u32));
		cudaMemcpy(d_a, anchor_points.data(), n_vars*sizeof(u32), cudaMemcpyHostToDevice);

		std::vector<u32> reference(n_vars*n_values);
		for(size_t i = 0; i < n_values; i++){
			for(size_t j = 0; j < n_vars; j++){
				reference.at(i + j*n_values) = nmod_pow_ui(anchor_points.at(j), i+1, mod);
			}
		}

		u32 *d_r;
		cudaMalloc(&d_r, n_vars*n_values*sizeof(u32));

		// Assuming that n_vars < n_values, this is the simplest way to divide the work
		dim3 threads_per_block(32, 1);
		INFO("threads per block = " + std::to_string(threads_per_block.x) + "," + std::to_string(threads_per_block.y));
		dim3 num_blocks( (n_values + threads_per_block.x - 1) / threads_per_block.x , n_vars);
		INFO("num blocks = " + std::to_string(num_blocks.x) + "," + std::to_string(num_blocks.y));
		evaluate_powers<<<num_blocks, threads_per_block>>>(n_vars, n_values, d_a, d_r, p);

		std::vector<u32> result(n_vars*n_values);
		cudaMemcpy(result.data(), d_r, n_vars*n_values*sizeof(u32), cudaMemcpyDeviceToHost);

		for(size_t i = 0; i < reference.size(); i++){
			u32 idx = i % n_values;
			u32 jdx = i / n_values;
			INFO("a["+std::to_string(jdx)+"] = " + std::to_string(anchor_points.at(jdx)));
			INFO("exponent = " + std::to_string(idx + 1));
			REQUIRE(result.at(i) == reference.at(i));
		}

		cudaFree(d_a);
		cudaFree(d_r);
	}

	SECTION("Polynomial 1"){
		auto poly = "3 + x*y + x^2 + 4*y^3";

		//auto reconst_poly = zippel_reconstruct(poly, 2, 16);

	}
}
