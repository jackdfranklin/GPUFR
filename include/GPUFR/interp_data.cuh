#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>

#include "GPUFR/ff_math.cuh"
#include "GPUFR/ntt.cuh"

#define NTT_PRIMES "primes_roots_14.csv"

class interp_data
{
    private:
    int prime_id;
    std::vector<u32> primes;
    std::vector<u32*> dense_results;
    std::vector<std::string> variables;

    public:
    int two_exponent;
    int n_samps;
    int n_vars;
    int flat_size;

    interp_data(int max_power, int number_variables) : n_vars(number_variables)
    {
        two_exponent = 0;
        n_samps = (1<<two_exponent) + 1;
        while (n_samps < max_power)
        {
            two_exponent += 1;
            n_samps = (1<<two_exponent) + 1; 
        }

        flat_size = pow(n_samps, n_vars);
        prime_id = 0;
    }

    std::vector<u32> next_prime();

    void add_result(u32* probes, u32 prime);

    std::string to_str();

    ~interp_data()
    {
        for (int i=0; i<dense_results.size(); i++)
        {
            delete[] dense_results[i];
        }
    }
};