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

__global__ void solve_transposed_vandermonde(u32 n, u32 *v, u32 *c, u32 *f, u32 p, u32 *c_out){
	i64 idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	u32 t = 1;
	u32 b = 1;
	u32 s = f[n - 1];

	for(size_t j = n - 1; j > 0; j--){
		b = ff_add(c[j], ff_multiply(v[idx], b, p), p);
		s = ff_add(s, ff_multiply(f[j - 1], b, p), p);
		t = ff_add(ff_multiply(v[idx], t, p), b, p);
	}

	c_out[idx] = ff_divide(ff_divide(s, t, p), v[idx], p);
}

void interpolate(u32 n, u32 *d_v, u32 *d_f, u32 p, u32 *d_c_out){

	u32 *d_c;
	cudaMalloc(&d_c, n*sizeof(u32));
	extract_master_poly_coefficients(n, d_v, p, d_c);

	solve_transposed_vandermonde<<<1, n>>>(n, d_v, d_c, d_f, p, d_c_out);

	cudaFree(d_c);
} 
