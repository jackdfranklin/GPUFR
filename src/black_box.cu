#include "GPUFR/black_box.cuh"

BlackBox::BlackBox(const std::vector<std::string> &tokens_list, const std::vector<std::string> &var_labels, mpz_t coeff_bound) : variables(var_labels), tokens(tokens_list)
{
    long int exponent;
    double mantissa = mpz_get_d_2exp(&exponent, coeff_bound);
    double log2_x = log(mantissa) + log(2.0) * (double)exponent;
    num_primes = ceil(log2_x / 31);
}

size_t BlackBox::get_token_size()
{
    return tokens.size();
}

int BlackBox::get_num_primes()
{
    return num_primes;
}

void BlackBox::init_gpu(int probe_len)
{
    this->probe_len = probe_len;
    size_tokens = tokens.size()*num_primes;
    bytes_tokens = size_tokens*sizeof(cu_type::string<STRING_LEN>);
    bytes_var_labels = variables.size()*sizeof(cu_type::string<STRING_LEN>);

    stack_depth = get_max_depth(tokens);
    stack_allocation_size = stack_depth * probe_len * num_primes;
    bytes_stack_allocation = stack_allocation_size*sizeof(u32);

    CUDA_SAFE_CALL(cudaMalloc(&d_cu_tokens, bytes_tokens));
    CUDA_SAFE_CALL(cudaMalloc(&d_cu_var_labels, bytes_var_labels));
    CUDA_SAFE_CALL(cudaMalloc(&d_stack_allocation, bytes_stack_allocation));

    cu_tokens = new cu_type::string<10>[size_tokens];
}

void BlackBox::load_to_gpu(Context ctx)
{
    cu_var_labels = to_cu_string(variables);

    for (int i=0; i<num_primes; i++)
    {
        cu_type::string<10> *write_loc = cu_tokens + i*tokens.size();
        to_cu_string(tokens, variables, write_loc, ctx.get_prime(i));
    }

    CUDA_SAFE_CALL(cudaMemcpy(d_cu_tokens, cu_tokens, bytes_tokens, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_cu_var_labels, cu_var_labels, bytes_var_labels, cudaMemcpyHostToDevice));
}

void BlackBox::unload_from_gpu()
{
    CUDA_SAFE_CALL(cudaFree(d_cu_tokens));
    CUDA_SAFE_CALL(cudaFree(d_cu_var_labels));
    CUDA_SAFE_CALL(cudaFree(d_stack_allocation));
}

u32* BlackBox::get_stack_alloc(int id)
{
    u32* stack_alloc = d_stack_allocation + stack_depth*probe_len*id;
    return stack_alloc;
}

cu_type::string<10>* BlackBox::get_tokens(int id)
{
    cu_type::string<10>* tokens = d_cu_tokens + this->tokens.size()*id;
    return tokens;
}

BlackBox::~BlackBox()
{
    delete[] cu_tokens;
    delete[] cu_var_labels;
}
