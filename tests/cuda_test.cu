#include <catch2/catch_test_macros.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void test_kernel(bool *out_param){
	*out_param = true;
}

TEST_CASE("CUDA Test", "[CUDA]"){
	cudaSetDevice(0);

	bool *d_result;
	cudaMalloc(&d_result, sizeof(bool));

	test_kernel<<<1,1>>>(d_result);

	bool h_result = false;
	cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	REQUIRE(h_result == true);	
}

