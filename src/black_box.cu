#include "GPUFR/black_box.cuh"

BlackBox::BlackBox(const std::vector<std::string> &tokens_list, const std::vector<std::string> &var_labels) : variables(var_labels), tokens(tokens_list)
{
}

size_t BlackBox::get_token_size()
{
    return tokens.size();
}

void BlackBox::init_gpu(int probe_len)
{
    bytes_tokens = tokens.size()*sizeof(cu_type::string<STRING_LEN>);
    bytes_var_labels = variables.size()*sizeof(cu_type::string<STRING_LEN>);

    stack_depth = get_max_depth(tokens);
    stack_allocation_size = stack_depth * probe_len;
    bytes_stack_allocation = stack_allocation_size*sizeof(u32);

    CUDA_SAFE_CALL(cudaMalloc(&d_cu_tokens, bytes_tokens));
    CUDA_SAFE_CALL(cudaMalloc(&d_cu_var_labels, bytes_var_labels));
    CUDA_SAFE_CALL(cudaMalloc(&d_stack_allocation, bytes_stack_allocation));
}

void BlackBox::load_to_gpu(u32 prime)
{
    cu_tokens = to_cu_string(tokens, variables, prime);
    cu_var_labels = to_cu_string(variables);
    CUDA_SAFE_CALL(cudaMemcpy(d_cu_tokens, cu_tokens, bytes_tokens, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_cu_var_labels, cu_var_labels, bytes_var_labels, cudaMemcpyHostToDevice));
}

void BlackBox::unload_from_gpu()
{
    CUDA_SAFE_CALL(cudaFree(d_cu_tokens));
    CUDA_SAFE_CALL(cudaFree(d_cu_var_labels));
    CUDA_SAFE_CALL(cudaFree(d_stack_allocation));
}