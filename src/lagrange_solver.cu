#include "GPUFR/lagrange_solver.cuh"

// TODO: Make this dynamic
#define MAX_VARS 6 // The maximum number of variable to reconstruct over
#define MAX_EXPONENT 101 // The maximum exponent in the polynomeal
#define UNSIGNED_TYPE unsigned
#define PRIME 105097513 // Must be less than (max unsigend) / 2

__host__ __device__ int as_int(u32 val)
{
    int result = val;
    if (result > PRIME/2) result = result - PRIME;
    return result;
}

__host__ __device__ void print_vec(const u32* vec, int size)
{
    for (int i=0; i<size; i++)
    {
        printf("%i, ", as_int(vec[i]));
    }
    printf("\n");
}

__device__ u32 fun(u32 *vars)
{
    u32 result;
    result = vars[0];
    return result;
}

__global__ void compute_probes(const u32 *xs, u32 *probes, u32 *probes_2, int n_vars, int n_samps) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    u32 test_params[MAX_VARS];

    for (int j=0; j<n_vars; j++)
    {
        float ifloat = i;
        int dimension_index = static_cast<int>(floorf(ifloat/pow(n_samps, j))) % n_samps;//floorf((pow(n_samps, j)));
        test_params[j] = xs[j*n_samps+dimension_index];
    }
        
    probes[i] = fun(test_params);
    probes_2[i] = 0;
}


__device__ void atomic_add(u32 *l_val, u32 r_val)
{
    u32 assumed, old;
    u32 value;

    old = (*l_val);
    do
    {
        value = *l_val;
        assumed = old;
        old = atomicCAS(&((*l_val)), assumed, ff_add(value, r_val, PRIME));
    } while (assumed != old);
}

__device__ u32 compute_denom_nd(int current_index, const u32 *xs, int dim, int n_vars, int n_samps, int idx)
{
    int flat_current_index = dim*n_samps + static_cast<int>(floorf(idx/pow(n_samps, dim)))%n_samps;

    u32 denom;
    denom = 1;

    for (int i=0; i<n_samps; i++)
    {
        int flat_index = dim*n_samps + i;
        if (flat_index != flat_current_index)
        {
            denom = ff_multiply(denom, (ff_subtract(xs[flat_current_index],  xs[flat_index], PRIME)), PRIME);
        }
    }

    return denom;
}

__global__ void get_lagrange_coeffs_nd(const u32 *xs, u32 *ys, u32 *out, const u32 *lagrange, int dim, int n_vars, int n_samps, int two_exponent)
{
    // Computes the coefficients to the Lagrange polynomials and writes them to ys

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int pol_size = pow(2, two_exponent) + 1;
    int step_size = 2*(n_samps-1) - pol_size;

    // if (idx == 1){
    // printf("pol_size: %i \n", pol_size);
    // printf("step_size: %i \n", step_size);
    // }

    u32 denom;
    denom = compute_denom_nd(idx, xs, dim, n_vars, n_samps, idx);

    u32 coefficient = ff_divide(ys[idx], denom, PRIME);

    int index_step_ys = pow(n_samps, dim);
    int power = static_cast<int>(floorf(pow(n_samps, dim+1)));
    int start_index_ys = floorf(idx / power)*power + idx%index_step_ys;

    int index_xs = dim*n_samps + static_cast<int>(floorf(idx/pow(n_samps, dim)))%n_samps;

    for (int i=0; i<n_samps; i++)
    {
        int flat_index_ys = start_index_ys+i*index_step_ys;
        int flat_index_lagrange = index_xs*n_samps + i;
        int actual_index_lagrange = step_size*(flat_index_lagrange/pol_size) + flat_index_lagrange;
        // printf("idx: %i i: %i probe index: %i start index: %i to add: %i denom: %i lag: %i val: %i y: %i \n", idx, i, flat_index_ys, power, as_int(coefficient), as_int(denom), as_int(lagrange[flat_index_lagrange]), as_int(ff_multiply(ff_divide(lagrange[flat_index_lagrange], denom, PRIME), ys[idx], PRIME)), ys[idx]);
        u32 to_add = ff_multiply(ff_divide(lagrange[actual_index_lagrange], denom, PRIME), ys[idx], PRIME);
        atomic_add(&out[flat_index_ys], to_add); // TODO: more efficient reduction, this is the current bottleneck
    }

    ys[idx] = 0;

}


void convolve_cpp(const u32 *kernel, const u32 *signal, u32 *out, int kernel_size, int signal_size)
{
    int result_size = signal_size+kernel_size-1;
    int pad_size = kernel_size-1;
    for (int i=0; i<result_size; i++)
    {
        out[i] = 0;
        for (int j = 0; j < kernel_size; j++)
        {
            if (i+j >= pad_size && i+j-pad_size < signal_size)
            {
                out[i] = ff_add(out[i], ff_multiply(kernel[kernel_size - 1 - j], signal[i+j-pad_size], PRIME), PRIME);
            }
        }
    }
}

void compute_lagrange_pol(const u32 *xs, u32 *lagrange, int dim, int n_vars, int n_samps)
{
    u32 root_arr[2];
    u32 tmp[MAX_EXPONENT];
    u32 tmp2[MAX_EXPONENT];

    // Loop over each x to get l(x, xi)
    for (int i=0; i<n_samps; i++)
    {
        tmp[0] = 1.0;
        for (int k=1; k<n_samps; k++)
        {
            tmp[k] = 0.0;
        }


        // Iteratively convolve to compute expansion
        for (int j=0; j<n_samps; j++)
        {
            int x_index = dim*n_samps + j;

            if (i != j)
            {
                root_arr[0] = ff_subtract(0, xs[x_index], PRIME);
                root_arr[1] = 1;

                convolve_cpp(tmp, root_arr, tmp2, n_samps, 2);

                for (int k=0; k<n_samps; k++)
                {
                    tmp[k] = tmp2[k];
                }
            }
        }

        // Copy currnet expansion into lagrange
        for (int j=0; j<n_samps; j++)
        {
            int x_index = dim*n_samps + i;
            int lagrange_index = x_index*n_samps + j;

            lagrange[lagrange_index] = tmp[j];

            int pLag = tmp[j];
            if (pLag > PRIME/2) pLag = tmp[j] - PRIME;
            // printf("idx: %i term: %i expansion: %i \n", i, j, pLag);
        }
    }
}


std::string nd_poly_to_string_flat(const std::vector<double>& coef_flat, const std::vector<std::string>& variables, int n_samps) {
    // From chat GPT
    int dim = variables.size();
    std::ostringstream result;
    for (size_t i = 0; i < coef_flat.size(); ++i) {
        double c = coef_flat[i];
        if (sqrt(pow(c, 2)) >= 1) {
            c = c > PRIME/2.0 ? c-PRIME : c;
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

// __device__ void convolve_gpu(u32* kernel, u32* signal, u32* out, int out_start_loc, int in_size)
// {
//     int result_size = in_size+in_size-1;
//     int pad_size = in_size-1;
//     for (int i=0; i<result_size; i++)
//     {
//         int out_indx = out_start_loc+i;
//         out[out_indx] = 0;
//         for (int j = 0; j < in_size; j++)
//         {
//             if (i+j >= pad_size && i+j-pad_size < in_size)
//             {
//                 out[out_indx] = ff_add(out[out_indx], ff_multiply(kernel[in_size - j], signal[i+j-pad_size], PRIME), PRIME);
//             }
//         }
//     }
// }

__device__ void convolve_gpu(const u32 *kernel, const u32 *signal, u32 *out, int kernel_size, int signal_size, int out_start_loc)
{
    int result_size = signal_size+kernel_size-1;
    int pad_size = kernel_size-1;
    for (int i=0; i<result_size; i++)
    {
        int out_indx = out_start_loc+i;
        out[out_indx] = 0;
        for (int j = 0; j < kernel_size; j++)
        {
            if (i+j >= pad_size && i+j-pad_size < signal_size)
            {
                if (j == 0) {
                    // Account for padding
                    out[out_indx] = ff_add(out[out_indx], ff_multiply(0, signal[i+j-pad_size], PRIME), PRIME);
                } else {
                    out[out_indx] = ff_add(out[out_indx], ff_multiply(kernel[kernel_size - 1 - j], signal[i+j-pad_size], PRIME), PRIME);
                }
            }
        }
    }
}

__global__ void lagrange_convolution(u32 *lagrange, const u32 *lagrange_tmp, int level)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // if (idx == 0)
    // {
    //     printf("level: %i \n", level);
    // }

    int sub_pol_size = pow(2, level) + 1;

    int step_size = pow(2, level+2);
    int start_val_ker = idx*step_size;
    int start_val_sig = idx*step_size + step_size/2;

    int kernel_size = sub_pol_size;
    int signal_size = sub_pol_size;

    // For some ungodly reason instantiating the new pointer without const mutates the lagrange_tmp values!!!???!!!
    const u32 *kernel = lagrange_tmp+start_val_ker;
    const u32 *signal = lagrange_tmp+start_val_sig;
    // if (idx == 0)
    // {
    //     printf("0 level: %i start_val_ker: %i start_val_sig: %i \n", level, start_val_ker, start_val_sig);
    //     // print_vec(kernel, signal_size);
    //     // print_vec(signal, signal_size);
    //     print_vec(lagrange_tmp, 20);
    //     print_vec(lagrange, 20);
    // }
    
    // for (int i=0; i<sub_pol_size; i++)
    // {
    //     kernel[i] = lagrange[start_val_ker+i];
    //     signal[i] = lagrange[start_val_sig+i];
    // }

    if (idx == 0)
    {
        printf("level: %i start_val_ker: %i start_val_sig: %i \n", level, start_val_ker, start_val_sig);
        print_vec(kernel, signal_size);
        print_vec(signal, signal_size);
        print_vec(lagrange_tmp, 20);
        print_vec(lagrange, 20);
    }

    // kernel[sub_pol_size] = 0; // Padding for convolution this is causing an issue because the lagrange and tmp arrays become a diffeernet length

    int n_samps = pow(2, 6) + 1;
    int probe_len = pow(n_samps, 1);
    int lagrange_size = (n_samps-1)*n_samps*2;

    if (start_val_ker > lagrange_size)
    {
        printf("error \n");
        return;
    }
    if (start_val_ker < lagrange_size)
    {
        convolve_gpu(kernel, signal, lagrange, kernel_size, signal_size, start_val_ker);
    }

    if (idx == 0)
    {
        print_vec(lagrange, 20);
    }
}

__global__ void init_lagrange_branch_a(const u32* xs, u32* lagrange, u32* lagrange_tmp, int n_samps)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int sub_pol_size = 2;
    int start_index = idx*sub_pol_size;

    int read_index = idx%(n_samps-1);
    if (read_index < idx/(n_samps-1)) 
    {
        lagrange[start_index] = ff_subtract(0, xs[read_index], PRIME);
        lagrange[start_index+1] = 1;

        lagrange_tmp[start_index] = ff_subtract(0, xs[read_index], PRIME);
        lagrange_tmp[start_index+1] = 1;
    }
}

__global__ void init_lagrange_branch_b(const u32* xs, u32* lagrange, u32* lagrange_tmp, int n_samps)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int sub_pol_size = 2;
    int start_index = idx*sub_pol_size;

    int read_index = idx%(n_samps-1);
    if (read_index >= idx/(n_samps-1)) 
    {
        lagrange[start_index] = ff_subtract(0, xs[read_index + 1], PRIME);
        lagrange[start_index+1] = 1;

        lagrange_tmp[start_index] = ff_subtract(0, xs[read_index + 1], PRIME);
        lagrange_tmp[start_index+1] = 1;
    }
}

void multi_interp(int n_vars, int two_exponent)
{
    int n_samps = pow(2, two_exponent) + 1;
    int probe_len = pow(n_samps, n_vars);
    int lagrange_size = n_vars*(n_samps-1)*n_samps*2;
    u32 lagrange_polynomials[lagrange_size];

    u32 probes[probe_len];
    u32 xs[n_vars*n_samps];
    std::srand(time(0));

    for (int i=0; i<n_vars; i++)
    {
        for (int j=0; j<n_samps; j++)
        {
            int flat_index = i*n_samps + j;
            xs[flat_index] = (j+1)%PRIME;
            // xs[flat_index] = (std::rand())%PRIME;
        }
    }

    u32 *d_xs, *d_probes, *d_probes_2, *d_lagrange, *d_lagrange_tmp;

    // Size in bytes for each vector
    size_t bytes_xs = n_vars*n_samps * sizeof(u32);
    size_t bytes_probes = probe_len * sizeof(u32);
    size_t bytes_lagrange = lagrange_size * sizeof(u32);

    CUDA_SAFE_CALL(cudaSetDevice(1));

    // Allocate memory on the device
    CUDA_SAFE_CALL(cudaMalloc(&d_xs, bytes_xs));
    CUDA_SAFE_CALL(cudaMalloc(&d_probes, bytes_probes));
    CUDA_SAFE_CALL(cudaMalloc(&d_probes_2, bytes_probes));
    CUDA_SAFE_CALL(cudaMalloc(&d_lagrange, bytes_lagrange));
    CUDA_SAFE_CALL(cudaMalloc(&d_lagrange_tmp, bytes_lagrange));

    CUDA_SAFE_CALL(cudaMemcpy(d_xs, xs, bytes_xs, cudaMemcpyHostToDevice));

    int required_threads = probe_len;
    int threadsPerBlock = required_threads>256? 256 : probe_len;
    int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    // Computre all probes
    compute_probes<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_probes, d_probes_2, n_vars, n_samps);

    required_threads = lagrange_size;
    threadsPerBlock = required_threads>256? 256 : probe_len;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    // Dispatch together TODO
    init_lagrange_branch_a<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_lagrange, d_lagrange_tmp, n_samps);
    init_lagrange_branch_b<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_lagrange, d_lagrange_tmp, n_samps);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    for (int i=0; i<two_exponent; i++)
    {
        required_threads = pow(2, two_exponent-1-i);
        // required_threads = lagrange_size/pow(2, i+2);
        printf("required_threads: %i \n", required_threads);
        threadsPerBlock = required_threads>256? 256 : probe_len;
        blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

        lagrange_convolution<<<blocksPerGrid, threadsPerBlock>>>(d_lagrange, d_lagrange_tmp, i);
        std::swap(d_lagrange, d_lagrange_tmp);

        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    std::swap(d_lagrange, d_lagrange_tmp);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(lagrange_polynomials, d_lagrange_tmp, bytes_lagrange, cudaMemcpyDeviceToHost));



    // Perform multidimensional interpolation
    required_threads = probe_len;
    threadsPerBlock = required_threads>256? 256 : probe_len;
    blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;
    for (int i=0; i<n_vars; i++)
    {
        get_lagrange_coeffs_nd<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_probes, d_probes_2, d_lagrange, i, n_vars, n_samps, two_exponent);
        std::swap(d_probes, d_probes_2);
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());

    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(probes, d_probes, bytes_probes, cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(lagrange_polynomials, d_lagrange, bytes_lagrange, cudaMemcpyDeviceToHost));

    std::vector<double> probe_vec(probe_len);
    for (int i=0; i<probe_len; i++)
    {
        std::cout << "probe: " << probes[i] << " ";
        probe_vec[i] = probes[i];
    }

    for (int i=0; i<lagrange_size; i++)
    {
        if (i%(2*(n_samps-1)) == 0) {
            std::cout << std::endl;
        }
        std::cout << as_int(lagrange_polynomials[i]) << " ";

    }

    std::vector<std::string> vars = {"x", "y", "z"};
    std::string poly = nd_poly_to_string_flat(probe_vec, vars, n_samps);
    std::cout << std::endl << poly << std::endl;

    // Free memory on the device
    CUDA_SAFE_CALL(cudaFree(d_xs));
    CUDA_SAFE_CALL(cudaFree(d_probes));
    CUDA_SAFE_CALL(cudaFree(d_probes_2));
    CUDA_SAFE_CALL(cudaFree(d_lagrange));
    CUDA_SAFE_CALL(cudaFree(d_lagrange_tmp));

}

// int main()
// {
//     multi_interp(3, 6);
//     return 0;
// }