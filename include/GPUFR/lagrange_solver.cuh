#pragma once

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

#include <cuda.h>

#include "GPUFR/ff_math.cuh"
#include "GPUFR/cuda_safe_call.cuh"
#include "GPUFR/ntt.cuh"
#include "GPUFR/detokenize.cuh"
#include "GPUFR/parser.hpp"
#include "GPUFR/interp_data.cuh"

__global__ void compute_probes_tokens(u32* stack_allocation, const cu_type::string<STRING_LEN>* tokens, const cu_type::string<STRING_LEN>* var_labels, int token_len, const u32 *xs, u32 *probes, u32 *probes_2, size_t max_stack, int n_vars, int n_samps, u32 prime, int required_threads);

__global__ void init_lagrange_branch_a(const u32* xs, u32* lagrange, u32* denom_tmp, int n_samps, int n_vars, u32 prime, int required_threads);

__global__ void init_lagrange_branch_b(const u32* xs, u32* lagrange, u32* denom_tmp, int n_samps, int n_vars, u32 prime, int required_threads);

__global__ void element_multiply(u32* d_lagrange, int pol_size, u32 prime, int required_threads);

__global__ void compactify(u32 *lagrange, u32 *lagrange_tmp, int pol_size, int pol_container_size, int required_threads);

__global__ void reduce_denoms_level(u32* denoms_tmp, int n_samps, int n_vars, int level, u32 prime, int required_threads);

__global__ void copy_denoms(u32* denoms, u32* denoms_tmp, int stride, int required_threads);

void reduce_denoms(u32* denoms, u32* denoms_tmp, int n_samps, int n_vars, int prime);

__global__ void compute_sub_pols_nd(u32* lagrange, u32* lagrange_tmp, u32* denom, int n_samps, u32 prime, int required_threads);

__device__ int get_probe_read_index(int warp_id, int lane_id, int probe_step, int probe_step_large, int start_offset);

__device__ int get_lagrange_read_index(int warp_id, int lane_id, int n_samps, int warp_step, int start_offset);

__global__ void reduce_sum_kernel(u32 *lagrange, u32* probes, u32 *output_probes, int n_samps, int n_vars, int dim, int probe_step, int probe_step_large, int exponent, u32 prime, int required_threads);

void reduce_lagrange_nd(u32* lagrange, u32* lagrange_tmp, u32* denoms, u32* probes, u32* probes_tmp, int n_samps, int n_vars, int dim, u32 prime);

u32* interpolate_dense(const std::vector<std::string> &tokens, const std::vector<std::string> &var_labels, int two_exponent, const std::string &ntt_primes);

void interpolate_dense(interp_data &id, const std::vector<std::string> &tokens, const std::vector<std::string> &var_labels, const std::string &ntt_primes);

