#pragma once

#include "GPUFR/types.hpp"
#include "GPUFR/ff_math.cuh"
#include "GPUFR/polynomial.cuh"
#include "GPUFR/interp_data.cuh"
#include "GPUFR/context.cuh"
#include "GPUFR/black_box.cuh"

#include <flint/flint.h>
#include <flint/fmpz_poly.h>
#include <vector>

std::vector<u32> fast_1d_taylor(BlackBox& bb, int dim);

__global__ void init_probes(u32* d_probes, u32 base_root, u32 prime, int required_threads);
void initialise_roots_of_unity(u32* d_probes, int probe_len, int num_primes, Context& ctx);
__global__ void evaluate_probes(u32* d_probes, u32* stack_allocation, size_t max_stack, const cu_type::string<STRING_LEN>* tokens, const cu_type::string<STRING_LEN>* var_labels, int token_len, int prime, int required_threads);
void execute_bb(BlackBox& bb, u32* d_probes, int probe_len, Context& ctx);
void run_interp(u32* d_probes, u32* d_results, size_t poly_size, size_t num_primes, Context ctx);
