#include "GPUFR/lagrange_solver.cuh"
#include "GPUFR/bb_gen.hpp"
#include "GPUFR/nvrtc_helper.hpp"
#include "GPUFR/parser.hpp"

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>


// void test_pass(const std::vector<std::string> &tokens, const std::vector<std::string> &var_labels, int two_exponent, const std::string &ntt_primes, u32* results)
// {
//     int deviceCount = 0;
//     CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

//     cudaDeviceProp deviceProp;
//     cudaGetDeviceProperties(&deviceProp, 0);
//     if (deviceProp.concurrentKernels == 0) {
//         std::cerr << "GPU does not support concurrent kernel execution!" << std::endl;
//     }

//     int n_vars = var_labels.size();

//     int n_samps = pow(2, two_exponent) + 1;
//     int probe_len = pow(n_samps, n_vars);
//     int initial_pol_size = 4;
//     int lagrange_size = n_vars*(n_samps-1)*n_samps*initial_pol_size;
    
//     u32* lagrange_polynomials = new u32[lagrange_size];
//     u32* probes = new u32[probe_len];
//     u32* xs = new u32[n_vars*n_samps];
//     std::srand(time(0));

//     std::vector<u32> ws = get_w(ntt_primes, 0);
//     u32 prime = ws[0];

//     for (int i=0; i<n_vars; i++)
//     {
//         for (int j=0; j<n_samps; j++)
//         {
//             int flat_index = i*n_samps + j;
//             xs[flat_index] = flat_index%prime;
//         }
//     }

//     u32 *d_xs, *d_denoms, *d_probes, *d_probes_2, *d_lagrange, *d_lagrange_tmp;

//     // Size in bytes for each vector
//     size_t bytes_xs = n_vars*n_samps * sizeof(u32);
//     size_t bytes_denoms = n_vars*n_samps * sizeof(u32);
//     size_t bytes_probes = probe_len * sizeof(u32);
//     size_t bytes_lagrange = lagrange_size * sizeof(u32);

//     // Allocate memory on the device
//     CUDA_SAFE_CALL(cudaMalloc(&d_xs, bytes_xs));
//     CUDA_SAFE_CALL(cudaMalloc(&d_denoms, bytes_denoms));
//     CUDA_SAFE_CALL(cudaMalloc(&d_probes, bytes_probes));
//     CUDA_SAFE_CALL(cudaMalloc(&d_probes_2, bytes_probes));
//     CUDA_SAFE_CALL(cudaMalloc(&d_lagrange, bytes_lagrange));
//     CUDA_SAFE_CALL(cudaMalloc(&d_lagrange_tmp, bytes_lagrange));
//     CUDA_SAFE_CALL(cudaMemcpy(d_xs, xs, bytes_xs, cudaMemcpyHostToDevice));

//     int required_threads = probe_len;
//     int threadsPerBlock = required_threads>256? 256 : required_threads;
//     int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

//     // Computre all probes
//     cu_string<STRING_LEN>* cu_tokens = to_cu_string(tokens);
//     cu_string<STRING_LEN>* cu_var_labels = to_cu_string(var_labels);
//     cu_string<STRING_LEN> *d_cu_tokens, *d_cu_var_labels;

//     size_t bytes_tokens = tokens.size()*sizeof(cu_string<STRING_LEN>);
//     size_t bytes_var_labels = var_labels.size()*sizeof(cu_string<STRING_LEN>);

//     CUDA_SAFE_CALL(cudaMalloc(&d_cu_tokens, bytes_tokens));
//     CUDA_SAFE_CALL(cudaMalloc(&d_cu_var_labels, bytes_var_labels));

//     CUDA_SAFE_CALL(cudaMemcpy(d_cu_tokens, cu_tokens, bytes_tokens, cudaMemcpyHostToDevice));
//     CUDA_SAFE_CALL(cudaMemcpy(d_xs, xs, bytes_var_labels, cudaMemcpyHostToDevice));

//     compute_probes_tokens<<<blocksPerGrid, threadsPerBlock>>>(d_cu_tokens, d_cu_var_labels, tokens.size(), d_xs, d_probes, d_probes_2, n_vars, n_samps, prime, required_threads);

//     CUDA_SAFE_CALL(cudaFree(d_xs));
//     CUDA_SAFE_CALL(cudaFree(d_denoms));
//     CUDA_SAFE_CALL(cudaFree(d_probes));
//     CUDA_SAFE_CALL(cudaFree(d_probes_2));
//     CUDA_SAFE_CALL(cudaFree(d_lagrange));
//     CUDA_SAFE_CALL(cudaFree(d_lagrange_tmp));
//     CUDA_SAFE_CALL(cudaFree(d_cu_tokens));
//     CUDA_SAFE_CALL(cudaFree(d_cu_var_labels));

//     delete[] lagrange_polynomials;
//     delete[] cu_var_labels;
//     delete[] cu_tokens;
//     delete[] probes;
//     delete[] xs;
// }

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

    std::vector<std::string> tokens = parse_expression(fileContents);
    std::vector<std::string> var_lables = {"s", "t"};

    int n_vars = var_lables.size();
    int two_exp = 2;
    int n_samps = (1<<two_exp) + 1;
    int result_size = pow(n_samps, n_vars);
    u32* results = new u32[result_size];

    interpolate_dense(tokens, var_lables, two_exp, ntt_primes, results);

    delete[] results;
}