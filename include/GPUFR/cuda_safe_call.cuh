#pragma once
#ifdef __CUDACC__

#include <stdexcept> // std::runtime_error
#include <string> // std::string, std::to_string

#define CUDA_SAFE_CALL(err) { cuda_safe_call((err), __FILE__, __LINE__); }
struct cuda_error : public std::runtime_error { using std::runtime_error::runtime_error; };
inline void cuda_safe_call(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
        throw cuda_error(std::string(cudaGetErrorString(error)) + ": " + std::string(file) + " line " + std::to_string(line));
}
#else
#define CUDA_SAFE_CALL(err)
#endif