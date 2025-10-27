#pragma once

#include "GPUFR/stack.cuh"
#include "GPUFR/types.hpp"
#include "GPUFR/detokenize.cuh"
#include "GPUFR/cuda_safe_call.cuh"
#include "GPUFR/context.cuh"

#include <vector>
#include <gmp.h>

class BlackBox
{
    private:
    int num_primes;

    std::vector<std::string> variables;
    std::vector<std::string> tokens;

    cu_type::string<STRING_LEN>* cu_tokens;
    cu_type::string<STRING_LEN>* cu_var_labels;

    size_t probe_len;

    size_t bytes_tokens;
    size_t bytes_var_labels;

    size_t stack_allocation_size;
    size_t bytes_stack_allocation;

    public:
    size_t size_tokens;
    size_t stack_depth;

    cu_type::string<STRING_LEN> *d_cu_tokens, *d_cu_var_labels;
    u32 *d_stack_allocation;

    BlackBox(const std::vector<std::string> &tokens_list, const std::vector<std::string> &var_labels, mpz_t coeff_bound);
    ~BlackBox();

    int get_num_primes();
    u32* get_stack_alloc(int id);
    cu_type::string<10>* get_tokens(int id);

    size_t get_token_size();
    void init_gpu(int probe_len);
    void load_to_gpu(Context ctx);
    void unload_from_gpu();
};