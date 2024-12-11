#include "GPUFR/interp_data.cuh"

interp_data::interp_data(int max_power, const std::vector<std::string> &tokens_list, const std::vector<std::string> &var_labels) : variables(var_labels), tokens(tokens_list)
{
    two_exponent = 0;
    n_samps = (1<<two_exponent) + 1;
    n_vars = variables.size();
    n_tokens = tokens.size();
    while (n_samps <= max_power)
    {
        two_exponent += 1;
        n_samps = (1<<two_exponent) + 1; 
    }

    flat_size = pow(n_samps, n_vars);
    prime_id = 0;
}

// Returns a vector of the prime followed by successive roots of unity
std::vector<u32> interp_data::get_prime_roots()
{
    return std::vector<u32>(std::begin(precomp[prime_id]), std::end(precomp[prime_id]));
}

void interp_data::next_prime()
{
    prime_id += 1;
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

interp_data::~interp_data()
{
    for (int i=0; i<dense_results.size(); i++)
    {
        delete[] dense_results[i];
    }
}