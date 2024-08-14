#pragma once

#include "GPUFR/types.h"
#include "GPUFR/ff_math.cuh"

__device__ u32 master_poly_coefficient(u32 n, u32 i, u32 j, u32 *v, u32 *c, u32 p);

__global__ void master_poly_coefficient_iter(u32 n, u32 i, u32 *v, u32 *c_in, u32 p, u32 *c_out);

void extract_master_poly_coefficients(u32 n, u32 *v, u32 p, u32 *c_out);
