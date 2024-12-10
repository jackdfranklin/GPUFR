#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>

#include "GPUFR/ff_math.cuh"
#include "GPUFR/ntt.cuh"
#include "GPUFR/precomp.hpp"

#define NTT_PRIMES "primes_roots_14.csv"

class interp_data
{
    private:
    int prime_id;
    std::vector<u32> primes;
    std::vector<u32*> dense_results;
    std::vector<std::string> variables;
    std::vector<std::string> tokens;

    public:
    int two_exponent;
    int n_samps;
    int n_vars;
    int n_tokens;
    int flat_size;

    interp_data(int max_power, const std::vector<std::string> &tokens_list, const std::vector<std::string> &var_labels);

    std::vector<u32> next_prime();

    const std::vector<std::string>& get_tokens();
    
    const std::vector<std::string>& get_vars();

    void add_result(u32* probes, u32 prime);

    std::string to_str();

    ~interp_data();
};