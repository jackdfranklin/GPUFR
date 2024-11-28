#include "GPUFR/lagrange_solver.cuh"

// TODO: Make this dynamic
#define MAX_VARS 6 // The maximum number of variable to reconstruct over
#define MAX_EXPONENT 101 // The maximum exponent in the polynomeal
#define UNSIGNED_TYPE unsigned
// #define PRIME 105097513 // Must be less than (max unsigend) / 2

__host__ __device__ int as_int(u32 val, u32 prime)
{
    int result = val;
    if (result > prime/2) result = result - prime;
    return result;
}

__host__ __device__ void print_vec(const u32* vec, int size, u32 prime)
{
    for (int i=0; i<size; i++)
    {
        printf("%i, ", as_int(vec[i], prime));
    }
    // printf("\n");
    // printf("\n");
}

__device__ u32 fun(u32 *vars, u32 prime)
{
    u32 result;
    u32 x = *vars;
    // u32 y = *(vars + 1);
    // result = ff_pow(x, 2, prime) + 2;
    for (int i=0; i<(1<<12); i++)
        {
            result = ff_add(result, ff_multiply(i, ff_pow(x, i, prime), prime), prime);
        }
    return result;
}

__global__ void compute_probes(const u32 *xs, u32 *probes, u32 *probes_2, int n_vars, int n_samps, u32 prime, int required_threads) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < required_threads)
    {
        u32 test_params[MAX_VARS];

        for (int j=0; j<n_vars; j++)
        {
            float ifloat = i;
            int dimension_index = static_cast<int>(floorf(ifloat/pow(n_samps, j))) % n_samps;//floorf((pow(n_samps, j)));
            test_params[j] = xs[j*n_samps+dimension_index];
        }
            
        probes[i] = fun(test_params, prime);
        probes_2[i] = 0;
    }
}

__global__ void compute_probes_tokens(const cu_string<STRING_LEN>* tokens, const cu_string<STRING_LEN>* var_labels, int token_len, const u32 *xs, u32 *probes, u32 *probes_2, int n_vars, int n_samps, u32 prime, int required_threads) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < required_threads)
    {
        u32 test_params[MAX_VARS];

        for (int j=0; j<n_vars; j++)
        {
            float ifloat = i;
            int dimension_index = static_cast<int>(floorf(ifloat/pow(n_samps, j))) % n_samps;//floorf((pow(n_samps, j)));
            test_params[j] = xs[j*n_samps+dimension_index];
        }
            
        probes[i] = detokenize(tokens, var_labels, token_len, n_vars, test_params, prime);
        probes_2[i] = 0;
    }
}

__device__ void atomic_add(u32 *l_val, u32 r_val, u32 prime)
{
    u32 assumed, old;
    u32 value;

    old = (*l_val);
    do
    {
        value = *l_val;
        assumed = old;
        old = atomicCAS(&((*l_val)), assumed, ff_add(value, r_val, prime));
    } while (assumed != old);
}

__device__ u32 compute_denom_nd(int current_index, const u32 *xs, int dim, int n_vars, int n_samps, int idx, u32 prime)
{
    int flat_current_index = dim*n_samps + (idx/static_cast<int>(pow(n_samps, dim)))%n_samps;

    u32 denom;
    denom = 1;

    for (int i=0; i<n_samps; i++)
    {
        int flat_index = dim*n_samps + i;
        if (flat_index != flat_current_index) // Bad warp divergence ~3x slowdown
        {
            denom = ff_multiply(denom, (ff_subtract(xs[flat_current_index],  xs[flat_index], prime)), prime);
        }
    }

    return denom;
}

std::string nd_poly_to_string_flat(const std::vector<double>& coef_flat, const std::vector<std::string>& variables, int n_samps, u32 prime) {
    // From chat GPT
    int dim = variables.size();
    std::ostringstream result;
    for (size_t i = 0; i < coef_flat.size(); ++i) {
        double c = coef_flat[i];
        if (sqrt(pow(c, 2)) >= 1) {
            c = c > prime/2.0 ? c-prime : c;
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

__global__ void init_lagrange_branch_a(const u32* xs, u32* lagrange, u32* denom_tmp, int n_samps, int n_vars, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int sub_pol_size = 4;
        int start_index = idx*sub_pol_size;
        int large_step_size = (n_samps-1)*n_samps;
        int subtraction_index = idx/(n_samps-1);

        int read_index = idx%(n_samps-1) + (idx/large_step_size)*n_samps;
        if (read_index < idx/(n_samps-1)) 
        {
            lagrange[start_index] = ff_subtract(0, xs[read_index], prime);
            lagrange[start_index+1] = 1;
            lagrange[start_index+2] = 0;
            lagrange[start_index+3] = 0;

            denom_tmp[idx] = ff_subtract(xs[subtraction_index], xs[read_index], prime);
        }
    }
}

__global__ void init_lagrange_branch_b(const u32* xs, u32* lagrange, u32* denom_tmp, int n_samps, int n_vars, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int sub_pol_size = 4;
        int start_index = idx*sub_pol_size;
        int large_step_size = (n_samps-1)*n_samps;
        int subtraction_index = idx/(n_samps-1);

        int read_index = idx%(n_samps-1) + (idx/large_step_size)*n_samps;
        if (read_index >= idx/(n_samps-1)) 
        {
            lagrange[start_index] = ff_subtract(0, xs[read_index + 1], prime);
            lagrange[start_index+1] = 1;
            lagrange[start_index+2] = 0;
            lagrange[start_index+3] = 0;

            denom_tmp[idx] = ff_subtract(xs[subtraction_index], xs[read_index+1], prime);
        }
    }
}


__global__ void element_multiply(u32* d_lagrange, int pol_size, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int sub_pol_index = idx%pol_size;
        int pol_start_index = (idx/pol_size)*2*pol_size;

        u32 in_1 = d_lagrange[pol_start_index+sub_pol_index];
        u32 in_2 = d_lagrange[pol_start_index+pol_size+sub_pol_index];

        u32 out = ff_multiply(in_1, in_2, prime);

        d_lagrange[pol_start_index+sub_pol_index] = out;
        d_lagrange[pol_start_index+pol_size+sub_pol_index] = 0;
    }
}

__global__ void compactify(u32 *lagrange, u32 *lagrange_tmp, int pol_size, int pol_container_size, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int read_index = (idx/pol_size)*pol_container_size + idx%pol_size;
        // printf("idx %i read_index %i pol_container_size %i\n", idx, read_index, pol_container_size);

        lagrange_tmp[idx] = lagrange[read_index];
    }
}

__global__ void reduce_denoms_level(u32* denoms_tmp, int n_samps, int n_vars, int level, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int stride = 1<<(level+1);
        int sub_step = 1<<(level);
        int start_index = stride*idx;

        u32 in_1 = denoms_tmp[start_index];
        u32 in_2 = denoms_tmp[start_index+sub_step];

        denoms_tmp[start_index] = ff_multiply(in_1, in_2, prime);
    }
}

__global__ void copy_denoms(u32* denoms, u32* denoms_tmp, int stride, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int start_index = stride*idx;

        u32 in_1 = denoms_tmp[start_index];

        denoms[idx] = in_1;
    }
}

void reduce_denoms(u32* denoms, u32* denoms_tmp, int n_samps, int n_vars, int prime)
{
    int iterations = log2(n_samps-1);
    int required_threads, threadsPerBlock, blocksPerGrid;

    for (int i=0; i<iterations; i++)
    {
        required_threads = (1<<(iterations-i-1))*n_samps*n_vars;
        threadsPerBlock = required_threads>256? 256 : required_threads;
        blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

        reduce_denoms_level<<<blocksPerGrid, threadsPerBlock>>>(denoms_tmp, n_samps, n_vars, i, prime, required_threads);
    }

    required_threads = n_samps*n_vars;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    int stride = (n_samps-1);

    copy_denoms<<<blocksPerGrid, threadsPerBlock>>>(denoms, denoms_tmp, stride, required_threads);
}

__global__ void compute_sub_pols(u32* lagrange, u32* lagrange_tmp, u32* denom, u32* probes, int probe_stride, int n_samps, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int probe_index = idx/(n_samps);

        lagrange_tmp[idx] = ff_multiply(probes[probe_index*probe_stride], ff_divide(lagrange[idx], denom[probe_index], prime), prime);
    }
}

__global__ void compute_sub_pols_nd(u32* lagrange, u32* lagrange_tmp, u32* denom, int n_samps, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int probe_index = idx/(n_samps);

        lagrange_tmp[idx] = ff_divide(lagrange[idx], denom[probe_index], prime);
    }
}

__global__ void reduce_lagrange_level(u32* lagrange, int n_samps, int n_vars, int level, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        // int sub_step = 1<<(level)*n_samps;
        int stride = (level+1)*n_samps;
        int start_index = idx%n_samps + (idx/n_samps)*(2*stride);

        // int over_stride = (level+1)*n_samps;

        u32 in_1 = lagrange[start_index];
        u32 in_2 = lagrange[start_index+stride];

        printf("level: %i in1: %i in2: %i start_index: %i, start_index+sub_step: %i \n", level, in_1, in_2, start_index, start_index+stride);

        lagrange[start_index] = ff_add(in_1, in_2, prime);
    }
}

__global__ void reduce_lagrange_final(u32* lagrange, u32* probes, int probe_stride, int n_samps, u32 prime, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int sub_step = n_samps;

        u32 in_1 = lagrange[idx];
        u32 in_2 = lagrange[idx+sub_step];

        probes[idx*probe_stride] = ff_add(in_1, in_2, prime);
    }
}


__device__ int get_probe_read_index(int warp_id, int lane_id, int probe_step, int probe_step_large, int start_offset)
{
    int probe_start = (warp_id/probe_step_large)*probe_step_large + warp_id%probe_step;
    int lane_probe = probe_start + (lane_id+start_offset)*probe_step;

    return lane_probe;
}

__device__ int get_lagrange_read_index(int warp_id, int lane_id, int n_samps, int warp_step, int start_offset)
{
    int lagrange_id = (warp_id/warp_step)%n_samps + (lane_id+start_offset)*n_samps;

    return lagrange_id;
}


// Each sum gets at least one warp
// When the sum us stored to probes, the extra probe is computed. This way everything can work in powers of 2
__global__ void reduce_sum_kernel(u32 *lagrange, u32* probes, u32 *output_probes, int n_samps, int n_vars, int dim, int probe_step, int probe_step_large, int exponent, u32 prime, int required_threads) {
    __shared__ u32 shared_data[32];  // Shared memory for inter-warp reduction, assuming max 32 warps per block

    int warp_size_mask = n_samps<warpSize? n_samps-1 : warpSize;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = tid % warp_size_mask;       // Lane within the warp
    int warp_id = (tid / warp_size_mask)%(blockDim.x/warp_size_mask);    // Warp ID within the block

    int block_size = (n_samps-1)<blockDim.x? (n_samps-1) : blockDim.x;
    int mask_lane_id = tid % block_size;
    int mask_warp_id = tid / block_size;

    int total_reductions = exponent<10? exponent: 10; // Assuming a box size of 1024 = 2^10 threds
    int completed_reductions = 0;
    int sum_step = 1;

    u32 sum = 0;

    // if (tid < required_threads)
    {
        // Sum excess points over the threads per block
        for (int i=0; i*blockDim.x<n_samps-1; i++)
        {
            int probe_index = get_probe_read_index(mask_warp_id, mask_lane_id+i*blockDim.x, probe_step, probe_step_large, 1);
            int lagrange_index = get_lagrange_read_index(mask_warp_id, mask_lane_id+i*blockDim.x, n_samps, probe_step, 1);

            sum = ff_add(sum, ff_multiply(lagrange[lagrange_index], probes[probe_index], prime), prime);
        }

        // Perform warp reduction; warp size is 32=2^5 so stop here
        for (;completed_reductions<total_reductions && completed_reductions<5; completed_reductions++) {
            sum_step = (1<<completed_reductions);
            sum = ff_add(sum, __shfl_down_sync(0xFFFFFFFF, sum, sum_step), prime);
        }

        // Save warp sum to shared
        if (lane_id == 0)
        {
            shared_data[warp_id] = sum;
        }
        __syncthreads();

        // Loads into the lanes in the first warp NB wont work if the threads per block > lanes per warp
        if (warp_id == 0)
        {
            sum = shared_data[lane_id];
        }


        // Intra block sum
        for (int i=0; i<total_reductions-completed_reductions; i++)
        {
            if (warp_id == 0)
            {
                sum_step = (1<<i);
                sum = ff_add(sum, __shfl_down_sync(0xFFFFFFFF, sum, sum_step), prime);
            }
        }

        // Copy back into relevent warps
        if (warp_id == 0)
        {
            shared_data[lane_id] = sum;
        }
        __syncthreads();
        if (lane_id == 0)
        {
            sum = shared_data[warp_id];
        }

        // Each warp adds the final odd term and saves to the appropriate probe
        if (lane_id == 0 && mask_lane_id == 0)
        {
            int probe_index = get_probe_read_index(mask_warp_id, mask_lane_id, probe_step, probe_step_large, 0);
            int lagrange_index = get_lagrange_read_index(mask_warp_id, mask_lane_id, n_samps, probe_step, 0);

            sum = ff_add(sum, ff_multiply(lagrange[lagrange_index], probes[probe_index], prime), prime);
            output_probes[mask_warp_id] = sum;
        }
    }
}

// Fills in a row in the new probe matrix by reducing the sum of lagrange polynomials to a single polynomial
void reduce_lagrange(u32* lagrange, u32* lagrange_tmp, u32* denoms, u32* probes, int probe_stride, int n_samps, int n_vars, u32 prime)
{
    int iterations = log2(n_samps-1);

    int required_threads, threadsPerBlock, blocksPerGrid;

    // Start with second polynomial to give a power of 2, add on first at the end
    u32* lagrange_tmp_even = lagrange_tmp+n_samps;

    required_threads = (n_samps)*n_samps;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    u32* lagrange_polynomials = new u32[required_threads];

    compute_sub_pols<<<blocksPerGrid, threadsPerBlock>>>(lagrange, lagrange_tmp, denoms, probes, probe_stride, n_samps, prime, required_threads);

    for (int i=0; i<iterations; i++)
    {
        required_threads = (1<<(iterations-i-1))*n_samps;
        threadsPerBlock = required_threads>256? 256 : required_threads;
        blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

        reduce_lagrange_level<<<blocksPerGrid, threadsPerBlock>>>(lagrange_tmp_even, n_samps, n_vars, i, prime, required_threads);
    }

    reduce_lagrange_final<<<blocksPerGrid, threadsPerBlock>>>(lagrange_tmp, probes, probe_stride, n_samps, prime, required_threads);
}

// TODO: make required threads a long
// Fills in a row in the new probe matrix by reducing the sum of lagrange polynomials to a single polynomial
void reduce_lagrange_nd(u32* lagrange, u32* lagrange_tmp, u32* denoms, u32* probes, u32* probes_tmp, int n_samps, int n_vars, int dim, u32 prime)
{
    int iterations = log2(n_samps-1);

    int required_threads, threadsPerBlock, blocksPerGrid;

    required_threads = (n_samps)*n_samps;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    u32* lagrange_polynomials = new u32[required_threads];

    compute_sub_pols_nd<<<blocksPerGrid, threadsPerBlock>>>(lagrange, lagrange_tmp, denoms, n_samps, prime, required_threads);

    int defaultThreadsPerBlock = 1024;
    required_threads = (n_samps-1)<defaultThreadsPerBlock? (n_samps-1)*pow(n_samps, n_vars) : defaultThreadsPerBlock*pow(n_samps, n_vars);
    threadsPerBlock = required_threads>defaultThreadsPerBlock? defaultThreadsPerBlock : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    int probe_step = pow(n_samps, dim);
    int probe_step_large = pow(n_samps, dim+1);

    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(lagrange_tmp, probes, probes_tmp, n_samps, n_vars, dim, probe_step, probe_step_large, iterations, prime, required_threads);
}


// TDOD: either switch to Karatsuba algorithm or FFT use Barett algorithm for division
void multi_interp(int n_vars, int two_exponent, const std::string &ntt_primes)
{
    int deviceCount = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.concurrentKernels == 0) {
        std::cerr << "GPU does not support concurrent kernel execution!" << std::endl;
    }

    int n_samps = pow(2, two_exponent) + 1;
    int probe_len = pow(n_samps, n_vars);
    int initial_pol_size = 4;
    int lagrange_size = n_vars*(n_samps-1)*n_samps*initial_pol_size;
    
    u32* lagrange_polynomials = new u32[lagrange_size];
    u32* probes = new u32[probe_len];
    u32* xs = new u32[n_vars*n_samps];
    std::srand(time(0));

    std::vector<u32> ws = get_w(ntt_primes, 0);
    u32 prime = ws[0];

    for (int i=0; i<n_vars; i++)
    {
        for (int j=0; j<n_samps; j++)
        {
            int flat_index = i*n_samps + j;
            xs[flat_index] = (std::rand())%prime;
        }
    }

    u32 *d_xs, *d_denoms, *d_probes, *d_probes_2, *d_lagrange, *d_lagrange_tmp;

    // Size in bytes for each vector
    size_t bytes_xs = n_vars*n_samps * sizeof(u32);
    size_t bytes_denoms = n_vars*n_samps * sizeof(u32);
    size_t bytes_probes = probe_len * sizeof(u32);
    size_t bytes_lagrange = lagrange_size * sizeof(u32);

    // Allocate memory on the device
    CUDA_SAFE_CALL(cudaMalloc(&d_xs, bytes_xs));
    CUDA_SAFE_CALL(cudaMalloc(&d_denoms, bytes_denoms));
    CUDA_SAFE_CALL(cudaMalloc(&d_probes, bytes_probes));
    CUDA_SAFE_CALL(cudaMalloc(&d_probes_2, bytes_probes));
    CUDA_SAFE_CALL(cudaMalloc(&d_lagrange, bytes_lagrange));
    CUDA_SAFE_CALL(cudaMalloc(&d_lagrange_tmp, bytes_lagrange));
    CUDA_SAFE_CALL(cudaMemcpy(d_xs, xs, bytes_xs, cudaMemcpyHostToDevice));

    int required_threads = probe_len;
    int threadsPerBlock = required_threads>256? 256 : required_threads;
    int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    // Computre all probes
    compute_probes<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_probes, d_probes_2, n_vars, n_samps, prime, required_threads);

    required_threads = lagrange_size/initial_pol_size;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    // Dispatch together TODO
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    init_lagrange_branch_a<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_xs, d_lagrange, d_lagrange_tmp, n_samps, n_vars, prime, required_threads);
    init_lagrange_branch_b<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_xs, d_lagrange, d_lagrange_tmp, n_samps, n_vars, prime, required_threads);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    reduce_denoms(d_denoms, d_lagrange_tmp, n_samps, n_vars, prime);

    for (int i=0; i<two_exponent; i++)
    {
        required_threads = lagrange_size/2;
        threadsPerBlock = required_threads>256? 256 : required_threads;
        blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

        int pol_size = 1<<(i+2);

        do_bulk_ntt(d_lagrange, d_lagrange_tmp, n_samps, n_vars, i, ws, prime); // doesnt account for higher dimensions
        element_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_lagrange_tmp, pol_size, prime, required_threads);
        do_bulk_ntt(d_lagrange_tmp, d_lagrange, n_samps, n_vars, i, ws, prime, true);
    }

    int pol_size = n_samps;
    int pol_container_size = (1<<(two_exponent+2));
    required_threads = pol_size*n_samps*n_vars;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    compactify<<<blocksPerGrid, threadsPerBlock>>>(d_lagrange, d_lagrange_tmp, pol_size, pol_container_size, required_threads); // Inefficient but not that bad
    std::swap(d_lagrange, d_lagrange_tmp);

    // Perform multidimensional interpolation
    required_threads = probe_len;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;
    for (int i=0; i<n_vars; i++)
    {
        int lagrange_sub_start = n_samps*n_samps*i;
        int denoms_sub_start = n_samps*i;

        u32 *d_lagrange_sub = d_lagrange+lagrange_sub_start;
        u32 *d_lagrange_tmp_sub = d_lagrange+lagrange_sub_start;
        u32 *d_denoms_sub = d_denoms+denoms_sub_start;

        reduce_lagrange_nd(d_lagrange_sub, d_lagrange_tmp_sub, d_denoms_sub, d_probes, d_probes_2, n_samps, n_vars, i, prime);
        
        std::swap(d_probes, d_probes_2);
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(probes, d_probes, bytes_probes, cudaMemcpyDeviceToHost));

    std::vector<double> probe_vec(probe_len);
    for (int i=0; i<probe_len; i++)
    {
        probe_vec[i] = probes[i];
    }

    std::vector<std::string> vars = {"x", "y", "z"};
    std::string poly = nd_poly_to_string_flat(probe_vec, vars, n_samps, prime);
    std::cout << std::endl << poly << std::endl;

    // Free memory on the device
    CUDA_SAFE_CALL(cudaFree(d_xs));
    CUDA_SAFE_CALL(cudaFree(d_denoms));
    CUDA_SAFE_CALL(cudaFree(d_probes));
    CUDA_SAFE_CALL(cudaFree(d_probes_2));
    CUDA_SAFE_CALL(cudaFree(d_lagrange));
    CUDA_SAFE_CALL(cudaFree(d_lagrange_tmp));

    delete[] lagrange_polynomials;
    delete[] probes;
    delete[] xs;
}

// TDOD: either switch to Karatsuba algorithm or FFT use Barett algorithm for division
void interpolate_dense(const std::vector<std::string> &tokens, const std::vector<std::string> &var_labels, int two_exponent, const std::string &ntt_primes, u32* results)
{
    int deviceCount = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.concurrentKernels == 0) {
        std::cerr << "GPU does not support concurrent kernel execution!" << std::endl;
    }

    int n_vars = var_labels.size();

    int n_samps = pow(2, two_exponent) + 1;
    int probe_len = pow(n_samps, n_vars);
    int initial_pol_size = 4;
    int lagrange_size = n_vars*(n_samps-1)*n_samps*initial_pol_size;
    
    u32* lagrange_polynomials = new u32[lagrange_size];
    u32* probes = new u32[probe_len];
    u32* xs = new u32[n_vars*n_samps];
    std::srand(time(0));

    std::vector<u32> ws = get_w(ntt_primes, 0);
    u32 prime = ws[0];

    for (int i=0; i<n_vars; i++)
    {
        for (int j=0; j<n_samps; j++)
        {
            int flat_index = i*n_samps + j;
            xs[flat_index] = (std::rand())%prime;
        }
    }

    u32 *d_xs, *d_denoms, *d_probes, *d_probes_2, *d_lagrange, *d_lagrange_tmp;

    // Size in bytes for each vector
    size_t bytes_xs = n_vars*n_samps * sizeof(u32);
    size_t bytes_denoms = n_vars*n_samps * sizeof(u32);
    size_t bytes_probes = probe_len * sizeof(u32);
    size_t bytes_lagrange = lagrange_size * sizeof(u32);

    // Allocate memory on the device
    CUDA_SAFE_CALL(cudaMalloc(&d_xs, bytes_xs));
    CUDA_SAFE_CALL(cudaMalloc(&d_denoms, bytes_denoms));
    CUDA_SAFE_CALL(cudaMalloc(&d_probes, bytes_probes));
    CUDA_SAFE_CALL(cudaMalloc(&d_probes_2, bytes_probes));
    CUDA_SAFE_CALL(cudaMalloc(&d_lagrange, bytes_lagrange));
    CUDA_SAFE_CALL(cudaMalloc(&d_lagrange_tmp, bytes_lagrange));
    CUDA_SAFE_CALL(cudaMemcpy(d_xs, xs, bytes_xs, cudaMemcpyHostToDevice));

    int required_threads = probe_len;
    int threadsPerBlock = required_threads>256? 256 : required_threads;
    int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    // Computre all probes
    cu_string<STRING_LEN>* cu_tokens = to_cu_string(tokens);
    cu_string<STRING_LEN>* cu_var_labels = to_cu_string(var_labels);
    cu_string<STRING_LEN>* d_cu_tokens, d_cu_var_labels;
    CUDA_SAFE_CALL(cudaMalloc(&d_cu_tokens, tokens.size()*sizeof(cu_string<STRING_LEN>)));
    CUDA_SAFE_CALL(cudaMalloc(&d_cu_var_labels, var_labels.size()*sizeof(cu_string<STRING_LEN>)));

    compute_probes_tokens<<<blocksPerGrid, threadsPerBlock>>>(tokens, var_labels, d_xs, d_probes, d_probes_2, n_vars, n_samps, prime, required_threads);

    required_threads = lagrange_size/initial_pol_size;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    // Dispatch together TODO
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    init_lagrange_branch_a<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_xs, d_lagrange, d_lagrange_tmp, n_samps, n_vars, prime, required_threads);
    init_lagrange_branch_b<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_xs, d_lagrange, d_lagrange_tmp, n_samps, n_vars, prime, required_threads);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    reduce_denoms(d_denoms, d_lagrange_tmp, n_samps, n_vars, prime);

    for (int i=0; i<two_exponent; i++)
    {
        required_threads = lagrange_size/2;
        threadsPerBlock = required_threads>256? 256 : required_threads;
        blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

        int pol_size = 1<<(i+2);

        do_bulk_ntt(d_lagrange, d_lagrange_tmp, n_samps, n_vars, i, ws, prime); // doesnt account for higher dimensions
        element_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_lagrange_tmp, pol_size, prime, required_threads);
        do_bulk_ntt(d_lagrange_tmp, d_lagrange, n_samps, n_vars, i, ws, prime, true);
    }

    int pol_size = n_samps;
    int pol_container_size = (1<<(two_exponent+2));
    required_threads = pol_size*n_samps*n_vars;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    compactify<<<blocksPerGrid, threadsPerBlock>>>(d_lagrange, d_lagrange_tmp, pol_size, pol_container_size, required_threads); // Inefficient but not that bad
    std::swap(d_lagrange, d_lagrange_tmp);

    // Perform multidimensional interpolation
    required_threads = probe_len;
    threadsPerBlock = required_threads>256? 256 : required_threads;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;
    for (int i=0; i<n_vars; i++)
    {
        int lagrange_sub_start = n_samps*n_samps*i;
        int denoms_sub_start = n_samps*i;

        u32 *d_lagrange_sub = d_lagrange+lagrange_sub_start;
        u32 *d_lagrange_tmp_sub = d_lagrange+lagrange_sub_start;
        u32 *d_denoms_sub = d_denoms+denoms_sub_start;

        reduce_lagrange_nd(d_lagrange_sub, d_lagrange_tmp_sub, d_denoms_sub, d_probes, d_probes_2, n_samps, n_vars, i, prime);
        
        std::swap(d_probes, d_probes_2);
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(results, d_probes, bytes_probes, cudaMemcpyDeviceToHost));

    std::vector<double> probe_vec(probe_len);
    for (int i=0; i<probe_len; i++)
    {
        probe_vec[i] = results[i];
    }

    std::vector<std::string> vars = {"x", "y", "z"};
    std::string poly = nd_poly_to_string_flat(probe_vec, vars, n_samps, prime);
    std::cout << std::endl << poly << std::endl;

    // Free memory on the device
    CUDA_SAFE_CALL(cudaFree(d_xs));
    CUDA_SAFE_CALL(cudaFree(d_denoms));
    CUDA_SAFE_CALL(cudaFree(d_probes));
    CUDA_SAFE_CALL(cudaFree(d_probes_2));
    CUDA_SAFE_CALL(cudaFree(d_lagrange));
    CUDA_SAFE_CALL(cudaFree(d_lagrange_tmp));

    delete[] lagrange_polynomials;
    delete[] cu_tokens;
    delete[] probes;
    delete[] xs;
}