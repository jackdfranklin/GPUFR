#include "GPUFR/vandermonde_solver.cuh"

__device__ u32 master_poly_coefficient(u32 n, u32 i, u32 j, u32 *v, u32 *c, u32 p){
	
	return ff_subtract(c[j], ff_multiply(v[i], c[j + 1], p), p);	
}

__global__ void init_coefficients(u32 n, u32 *v, u32 *c_out, u32 p){
	i64 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx == n - 1){
		c_out[idx] = ff_subtract(0, v[0], p);
	}
	else {
		c_out[idx] = 0;
	}
}

__global__ void master_poly_coefficient_iter(u32 n, u32 i, u32 *v, u32 *c_in, u32 p, u32 *c_out){
	
	i64 idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx == n-1){
		c_out[idx] = ff_subtract(c_in[idx], v[i], p);
	}
	else {
		c_out[idx] = master_poly_coefficient(n, i, idx, v, c_in, p);
	}
}

__global__ void copy_data(u32 *origin, u32 *destination){
	
	i64 idx = threadIdx.x + blockIdx.x * blockDim.x;

	destination[idx] = origin[idx];	
}

void extract_master_poly_coefficients(u32 n, u32 *v, u32 p, u32 *c_out){

	u32 *c_in;
	cudaMalloc(&c_in, n * sizeof(u32));

	init_coefficients<<<1, n>>>(n, v, c_in, p);	

	for(u32 i = 1; i < n; i++){
		master_poly_coefficient_iter<<<1,n>>>(n, i, v, c_in, p, c_out);
		copy_data<<<1,n>>>(c_out, c_in);
	}

	cudaFree(c_in);

}
