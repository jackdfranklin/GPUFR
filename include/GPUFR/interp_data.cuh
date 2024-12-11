#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>

#include "GPUFR/ff_math.cuh"
#include "GPUFR/ntt.cuh"
#include "GPUFR/precomp.hpp"
#include "GPUFR/parser.hpp"

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

    std::vector<u32> get_prime_roots();

    void next_prime();

    const std::vector<std::string>& get_tokens();

    const std::vector<std::string>& get_vars();

    void add_result(u32* probes, u32 prime);

    std::string to_str();

    ~interp_data();
};

// todo handle taking modulus in first instance to compute the tokens 
// have a function to increment the prime when ready
// make print statement work with mpz_t numbers
// Have a funciton to call the crt
