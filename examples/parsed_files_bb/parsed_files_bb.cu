#include "GPUFR/lagrange_solver.cuh"
#include "GPUFR/bb_gen.hpp"
#include "GPUFR/nvrtc_helper.hpp"
#include "GPUFR/parser.hpp"
#include "GPUFR/interp_data.cuh"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>

int main(int argc, char* argv[])
{
    auto start = std::chrono::high_resolution_clock::now();

    // std::ifstream file("/mt/home/jmaxwell/Documents/GPUFR/precomp/modded_funcs/hh_coeff1_n.txt"); // Replace with your file's path
    std::ifstream file("/mt/home/jmaxwell/Documents/GPUFR/examples/parsed_files_bb/simple_example.txt"); // Replace with your file's path
    // std::ifstream file("/mt/home/jmaxwell/Documents/GPUFR/examples/parsed_files_bb/example_fun.txt"); // Replace with your file's path
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
    }


    std::stringstream buffer;
    buffer << file.rdbuf();  // Read the entire file into the buffer
    std::string fileContents = buffer.str();  // Convert buffer to a string

    auto string_read = std::chrono::high_resolution_clock::now();


    std::vector<std::string> tokens = parse_expression(fileContents);

    auto parsed = std::chrono::high_resolution_clock::now();

    std::vector<std::string> var_lables = {"s", "t"};

    interp_data id(5, tokens, var_lables);

    auto pre_interp = std::chrono::high_resolution_clock::now();

    // for (int i=0; i<19; i++)
    {
        interpolate_dense(id);
    }

    auto post_interp = std::chrono::high_resolution_clock::now();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::chrono::duration<double> elapsed_read = string_read - start;
    std::chrono::duration<double> elapsed_parsed = parsed - string_read;
    std::chrono::duration<double> elapsed_interp = post_interp - pre_interp;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    std::cout << "Reading time: " << elapsed_read.count() << " seconds\n";
    std::cout << "Parsing time: " << elapsed_parsed.count() << " seconds\n";
    std::cout << "Interp time: " << elapsed_interp.count() << " seconds\n";

    std::cout << id.to_str() << std::endl;
}