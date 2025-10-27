//TODO
/*
1. We intend to interpolate functions of the form F(xi) = N(xi) / D(xi) where N and D are polynomials with integer coefficients.
2. This can be done by interpolating univariate rational functions over a grid of probes as with the Lagrange interpolation of polynomials.
3. Univariate rational interpolation can be done using a modified version of the extended Euclidean algorithm.
4. The algorithm is as follows:
    a. We wish to solve for Q(x) and P(x) such that R(x) Q(x) - P(x) = 0.
    b. We begin by determining the polynomial R(x) by lagrange interpolation over the probes within a chosen Galois field.
*/

#include "GPUFR/fast_taylor.cuh"

std::vector<u32> fast_1d_taylor(BlackBox& bb, int dim)
{
    int deviceCount = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    Context ctx = Context();

    int poly_size = pow(2, ceil(log2(dim)));

    u32 *d_probes, *d_results, *results;
    size_t bytes_probes = poly_size * sizeof(u32);
    CUDA_SAFE_CALL(cudaMalloc(&d_probes, bytes_probes));
    CUDA_SAFE_CALL(cudaMalloc(&d_results, bytes_probes));
    results = new u32[bytes_probes];

    bool probes_success = false;
    bb.init_gpu(poly_size);
    while (!probes_success){
        bb.load_to_gpu(ctx.get_prime());
        initialise_roots_of_unity(d_probes, poly_size, ctx);

        execute_bb(bb, d_probes, poly_size, probes_success, ctx);
        if (!probes_success)
            ctx.new_prime();
    }
    bb.unload_from_gpu();

    do_ntt(d_probes, d_results, poly_size, ctx.get_roots(), ctx.get_prime(), true);

    CUDA_SAFE_CALL(cudaMemcpy(results, d_results, bytes_probes, cudaMemcpyDeviceToHost));

    std::vector<u32> res = std::vector<u32>(results, results+poly_size);

    CUDA_SAFE_CALL(cudaFree(d_probes));
    CUDA_SAFE_CALL(cudaFree(d_results));

    delete[] results;
    return res;
}

__global__ void init_probes(u32* d_probes, u32 base_root, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        d_probes[idx] = ff_pow(base_root, idx, prime);
    }
}

void initialise_roots_of_unity(u32* d_probes, int probe_len, Context& ctx)
{
    int required_threads = probe_len;
    int threadsPerBlock = required_threads>256? 256 : required_threads;
    int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    u32 prime = ctx.get_prime();
    u32 base_root = ctx.get_root(log2(probe_len));

    init_probes<<<blocksPerGrid, threadsPerBlock>>>(d_probes, base_root, prime, required_threads);
}

__global__ void evaluate_probes(u32* d_probes, u32* stack_allocation, size_t max_stack, const cu_type::string<STRING_LEN>* tokens, const cu_type::string<STRING_LEN>* var_labels, int token_len, int prime, int required_threads)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < required_threads)
    {
        u32 test_params[1];
        test_params[0] = d_probes[idx];
            
        d_probes[idx] = detokenize(stack_allocation, tokens, var_labels, max_stack, idx, token_len, 1, test_params, prime);
    }
}

void execute_bb(BlackBox& bb, u32* d_probes, int probe_len, bool& probes_success, Context& ctx)
{
    int required_threads = probe_len;
    int threadsPerBlock = required_threads>256? 256 : required_threads;
    int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    evaluate_probes<<<blocksPerGrid, threadsPerBlock>>>(d_probes, bb.d_stack_allocation, bb.stack_depth, bb.d_cu_tokens, bb.d_cu_var_labels, bb.get_token_size(), ctx.get_prime(), required_threads);
    probes_success = true;
}