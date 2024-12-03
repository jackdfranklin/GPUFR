#pragma once

#include <vector>
#include <string>

#include "GPUFR/types.hpp"

int max_stack_depth(std::vector<std::string> &parsed_expression);

u32 evaluate_cpu(std::string &op, u32 L, u32 R);

void evaluate_gpu(size_t Nthreads, size_t Nblocks, std::string &op, u32* L, u32* R, u32 prime);
