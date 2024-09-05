#include "GPUFR/zippel.cuh"

#include "GPUFR/ff_math.cuh"

__global__ void evaluate_monomials(u32 n_variables, size_t pitch, u32 *anchor_points, u32 *exponents, u32 *result, u32 p){
	
	i64 idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	u32 r = 1;

	for(size_t i = 0; i < n_variables; i++){
		r = ff_multiply(r, ff_pow(anchor_points[i], exponents[idx + i*pitch], p), p);	
	}

	result[idx] = r;

}
 
