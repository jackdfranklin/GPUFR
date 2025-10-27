#pragma once

#include "GPUFR/types.hpp"
#include "GPUFR/ff_math.cuh"
#include "GPUFR/stack.cuh"
#include "GPUFR/string.cuh"
#include "GPUFR/parser.hpp"

#include <gmp.h>

#include <vector>
#include <string>
#include <stack>

#define STRING_LEN 10 // 10 chars to write max unsigned in denary

__device__ u32 detokenize(u32* stack_allocation, const cu_type::string<STRING_LEN>* tokens, const cu_type::string<STRING_LEN>* var_labels, int max_stack, int thread_id, int token_len, int n_vars, u32 *vars, u32 prime);

__device__ u32 to_u32(cu_type::string<STRING_LEN> &token, u32 *vars, const cu_type::string<STRING_LEN>* var_labels, int n_vars);

__device__ bool is_operator(const cu_type::string<STRING_LEN> &token);

__device__ u32 operator_to_function(const cu_type::string<STRING_LEN> &op, u32 L, u32 R, u32 prime);

__host__ cu_type::string<STRING_LEN>* to_cu_string(const std::vector<std::string> &tokens);

__host__ cu_type::string<STRING_LEN>* to_cu_string(const std::vector<std::string> &tokens, const std::vector<std::string> &vars, u32 prime);

__host__ void to_cu_string(const std::vector<std::string> &tokens, const std::vector<std::string> &vars, cu_type::string<STRING_LEN>* result, u32 prime);


