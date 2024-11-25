#include "GPUFR/lagrange_solver.cuh"
#include "GPUFR/bb_gen.hpp"
#include "GPUFR/nvrtc_helper.hpp"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char* argv[])
{
    std::string ntt_primes = argv[1];
    std::ifstream file("/mt/home/jmaxwell/Documents/GPUFR/examples/parsed_files_bb/example_fun.txt"); // Replace with your file's path
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();  // Read the entire file into the buffer
    std::string fileContents = buffer.str();  // Convert buffer to a string

	CUdevice cuda_device;
	CUcontext cuda_context;
	cuInit(0);
	cuDeviceGet(&cuda_device, 0);
	cuCtxCreate(&cuda_context, 0, cuda_device);	
	CUmodule module;

	gen_bb_module_nd(fileContents, std::vector<std::string> {"s", "t"}, cuda_context, cuda_device, module);

    int n_vars = 2;
    int n_samps = 4;
    int result_size = pow(n_samps, n_vars);
    u32* results = new u32[result_size];

    interpolate_dense(module, n_vars, n_samps, ntt_primes, results);

    delete[] results;
}