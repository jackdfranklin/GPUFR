#pragma once

#include "GPUFR/types.hpp"
#include "GPUFR/ff_math.cuh"

#include <vector>
#include <string>
#include <stack>


__device__ u32 detokenize(const std::vector<std::string> &tokens, const std::vector<std::string> &var_labels, u32 *vars, u32 prime);

__device__ u32 to_u32(std::string &token, u32 *vars, const std::vector<std::string> &var_labels);

__device__ bool is_operator(const std::string &token);

__device__ u32 operator_to_function(const std::string &op, u32 L, u32 R, u32 prime);