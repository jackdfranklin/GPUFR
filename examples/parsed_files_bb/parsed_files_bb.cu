#include "GPUFR/lagrange_solver.cuh"
#include "GPUFR/bb_gen.hpp"
#include "GPUFR/nvrtc_helper.hpp"
#include "GPUFR/parser.hpp"
#include "GPUFR/interp_data.cuh"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char* argv[])
{
    std::ifstream file("/mt/home/jmaxwell/Documents/GPUFR/precomp/modded_funcs/hh_coeff1_n.txt"); // Replace with your file's path
    // std::ifstream file("/mt/home/jmaxwell/Documents/GPUFR/examples/parsed_files_bb/example_fun.txt"); // Replace with your file's path
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();  // Read the entire file into the buffer
    std::string fileContents = buffer.str();  // Convert buffer to a string

    std::vector<std::string> tokens = parse_expression(fileContents);
    std::vector<std::string> var_lables = {"s", "t"};

    interp_data id(100, tokens, var_lables);

    for (int i=0; i<18; i++)
    {
        interpolate_dense(id);
    }


    std::cout << id.to_str() << std::endl;
}