#include "GPUFR/parser.hpp"
#include "GPUFR/types.hpp"
#include "GPUFR/nvrtc_helper.hpp"
#include "GPUFR/bb_gen.hpp"

#include <array>
#include <vector>
#include <deque>
#include <stack>
#include <algorithm>
#include <random>
#include <sstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <fstream>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <flint/flint.h>
#include <flint/nmod_types.h>
#include <flint/nmod.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

TEST_CASE("Parser File", "[Parsing][Evaluation]"){
	std::ifstream file("/mt/home/jmaxwell/Documents/GPUFR/precomp/modded_funcs/hh_coeff1_n_1000112129.txt"); // Replace with your file's path
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();  // Read the entire file into the buffer
    std::string fileContents = buffer.str();  // Convert buffer to a string

    std::cout << fileContents << std::endl;  // Output the file contents
	std::vector<std::string> rpn = parse_expression(fileContents);

	std::string parsed_expr = postfix_to_ff(rpn);
	// std::cout << parsed_expr << std::endl;
	REQUIRE(true);

	CUdevice cuda_device;
	CUcontext cuda_context;
	cuInit(0);
	cuDeviceGet(&cuda_device, 0);
	cuCtxCreate(&cuda_context, 0, cuda_device);	
	CUmodule module;

	gen_bb_module(fileContents, std::vector<std::string> {"s", "t"}, cuda_context, cuda_device, module);
}
