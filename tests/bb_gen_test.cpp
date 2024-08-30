#include "GPUFR/bb_gen.hpp"

#include "GPUFR/types.hpp"

#include <vector>
#include <string>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#include <flint/flint.h>
#include <flint/nmod_types.h>
#include <flint/nmod.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

TEST_CASE("Black Box Generation Test", "[Parsing][Evaluation]"){

	CUdevice cuda_device;
	CUcontext cuda_context;
	cuInit(0);
	cuDeviceGet(&cuda_device, 0);
	cuCtxCreate(&cuda_context, 0, cuda_device);	

	u32 p = GENERATE(13, 2546604103, 3998191247);
	nmod_t mod = {0};
	nmod_init(&mod, p);

	u32 number_of_values = 128;

	SECTION("Evaluations match with one variable"){

		const std::vector<std::string> vars = {"x"};
		const std::string expression = "3 + x^2 + 7*x^3";
		auto black_box = [mod](u32 x){
			u32 result = 3;
			result = nmod_add(result, nmod_pow_ui(x, 2, mod), mod);
			u32 temp = nmod_mul(7, nmod_pow_ui(x, 3, mod), mod);
			result = nmod_add(result, temp, mod);
			return result;
		};

		CUmodule module;
		gen_bb_module(expression, vars, cuda_context, cuda_device, module);
		CUfunction kernel;
		cuModuleGetFunction(&kernel, module, "evaluate");
		
		std::vector<u32> v = GENERATE_COPY(take(1, chunk(number_of_values, random((u32)0, p-1))));

		std::vector<u32> reference(number_of_values);
		std::transform(v.begin(), v.end(), reference.begin(), black_box);
		
		u32 *d_v, *d_c;

		cudaMalloc(&d_v, number_of_values*sizeof(u32));
		cudaMalloc(&d_c, number_of_values*sizeof(u32));

		cudaMemcpy(d_v, v.data(), number_of_values*sizeof(u32), cudaMemcpyHostToDevice);

		void *args[] = {&number_of_values, &d_v, &d_c, &p};

		cuLaunchKernel(kernel, 1, 1, 1, 128, 1, 1, 0, NULL, args, 0); 
		cuCtxSynchronize();

		std::vector<u32> result(number_of_values);
		cudaMemcpy(result.data(), d_c, number_of_values*sizeof(u32), cudaMemcpyDeviceToHost);

		for(size_t j = 0; j < number_of_values; j++){
			REQUIRE(result.at(j) == reference.at(j));
		}
	}

}
