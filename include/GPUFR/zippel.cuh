#pragma once

#include "GPUFR/types.hpp"

#include <vector>
#include <string>

std::vector<std::vector<u32>> zippel_reconstruct(std::string expression, u32 n_variables, u32 degree_bound);

__global__ void evaluate_monomials(u32 n_variables, u32 n_values, u32 *anchor_points, u32 *exponents, u32 *result, u32 p);


