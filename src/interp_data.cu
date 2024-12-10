#include "GPUFR/interp_data.cuh"

std::vector<u32> interp_data::next_prime()
{
    std::vector<u32> ws = std::vector<u32>(std::begin(precomp[prime_id]), std::end(precomp[prime_id]));
    prime_id += 1;
    return ws;
}

const std::vector<std::string>& interp_data::get_tokens()
{
    return tokens;
}

const std::vector<std::string>& interp_data::get_vars()
{
    return variables;
}

void interp_data::add_result(u32* probes, u32 prime)
{
    dense_results.emplace_back(probes);
    primes.emplace_back(prime);

    // Run CRT here
}

std::string interp_data::to_str()
{
    int dim = n_vars;
    std::ostringstream result;
    for (size_t i = 0; i < flat_size; ++i) {
        double c = dense_results[0][i];
        if (sqrt(pow(c, 2)) >= 1) {
            c = c > primes[0]/2.0 ? c-primes[0] : c;
            result << (c > 0 && result.tellp() > 0 ? "+ " : "") << std::fixed << std::setprecision(0) << c;
            for (int j = 0; j < dim; ++j) {
                int power = static_cast<int>(std::floor(i / std::pow(n_samps, j))) % n_samps;
                if (power > 0) {
                    result << "*" << variables[j] << "^" << power;
                }
            }
            result << " ";
        }
    }
    return result.str();
}