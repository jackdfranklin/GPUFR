#include "GPUFR/interp_data.cuh"

interp_data::interp_data(int max_power, const std::vector<std::string> &tokens_list, const std::vector<std::string> &var_labels) : variables(var_labels), tokens(tokens_list)
{
    is_mpz_init = false;
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
    if (!is_mpz_init)
    {
        mpz_init(crt_prime);
        mpz_set_ui(crt_prime, prime);
        dense_mpz = new mpz_t[flat_size];
        dense_mpz_tmp = new mpz_t[flat_size];
        for (int i=0; i<flat_size; i++)
        {
            mpz_init(dense_mpz[i]);
            mpz_init(dense_mpz_tmp[i]);
            mpz_set_ui(dense_mpz[i], probes[i]);
        }
        is_mpz_init = true;
    } else 
    {
        compute_crt(dense_mpz_tmp, dense_mpz, probes, crt_prime, prime, flat_size);
        mpz_mul_ui(crt_prime, crt_prime, prime);
        std::swap(dense_mpz_tmp, dense_mpz);
    }
}

std::string interp_data::to_str()
{
    int dim = n_vars;
    std::string str_coeff;
    bool is_zero;
    mpz_t zero;
    mpz_init(zero);
    mpz_set_ui(zero, 0);

    std::ostringstream result;
    for (size_t i = 0; i < flat_size; ++i) {
        char* c_str = mpz_get_str(nullptr, 10, dense_mpz[i]);  // Convert to a C-style string in base 10
        str_coeff = c_str; 
        is_zero = (mpz_cmp(dense_mpz[i], zero) == 0);
        if (!is_zero)
        {
            result << (result.tellp() > 0 ? "+ " : "") << str_coeff;
            for (int j = 0; j < dim; ++j) {
                int power = static_cast<int>(std::floor(i / std::pow(n_samps, j))) % n_samps;
                if (power > 0) {
                    result << "*" << variables[j] << "^" << power;
                }
            }
        result << " ";
        }
        // void (*freefunc)(void *, size_t);
        // mp_get_memory_functions(nullptr, nullptr, &freefunc);
        // freefunc(c_str, std::strlen(c_str) + 1);
    }


    return result.str();
}

interp_data::~interp_data()
{
    for (int i=0; i<flat_size; i++)
    {
        mpz_clear(dense_mpz[i]);
    }
    delete[] dense_mpz;
}