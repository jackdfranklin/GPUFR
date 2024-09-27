#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip> 

#include "GPUFR/ff_math.cuh"
#include "GPUFR/cuda_safe_call.cuh"

__device__ u32 fun(u32 *vars);

__global__ void compute_probes(const u32 *xs, u32 *probes, u32 *probes_2, int n_vars, int n_samps);

__device__ void atomic_add(u32 *l_val, u32 r_val);

__device__ u32 compute_denom_nd(int current_index, const u32 *xs, int dim, int n_vars, int n_samps, int idx);

__global__ void get_lagrange_coeffs_nd(const u32 *xs, u32 *ys, u32 *out, u32 *lagrange, int dim, int n_vars, int n_samps);

void convolve_cpp(const u32 *kernel, const u32 *signal, u32 *out, int kernel_size, int signal_size);

void compute_lagrange_pol(const u32 *xs, u32 *lagrange, int dim, int n_vars, int n_samps);

std::string nd_poly_to_string_flat(const std::vector<double>& coef_flat, const std::vector<std::string>& variables, int n_samps);

void multi_interp(int n_vars, int n_samps);
