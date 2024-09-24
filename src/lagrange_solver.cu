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

__global__ void get_lagrange_coeffs_nd(const u32 *xs, u32 *ys, u32 *out, u32 *lagrange, int dim, int n_vars, int n_samps)
{
    // Computes the coefficients to the Lagrange polynomials and writes them to ys

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
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
        // printf("idx: %i i: %i probe index: %i start index: %i to add: %i denom: %i lag: %i val: %i y: %i \n", idx, i, flat_index_ys, power, as_int(coefficient), as_int(denom), as_int(lagrange[flat_index_lagrange]), as_int(ff_multiply(ff_divide(lagrange[flat_index_lagrange], denom, PRIME), ys[idx], PRIME)), ys[idx]);
        atomic_add(&out[flat_index_ys], ff_multiply(ff_divide(lagrange[flat_index_lagrange], denom, PRIME), ys[idx], PRIME));
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


void multi_interp(int n_vars, int n_samps)
{
    int probe_len = pow(n_samps, n_vars);
    u32 lagrange_polynomials[n_vars*n_samps*n_samps];

    u32 probes[probe_len];
    u32 xs[n_vars*n_samps];
    std::srand(time(0));

    for (int i=0; i<n_vars; i++)
    {
        for (int j=0; j<n_samps; j++)
        {
            int flat_index = i*n_samps + j;
            xs[flat_index] = (std::rand())%PRIME;
        }
    }

    u32 *d_xs, *d_probes, *d_probes_2, *d_lagrange;

    // Size in bytes for each vector
    size_t bytes_xs = n_vars*n_samps * sizeof(u32);
    size_t bytes_probes = probe_len * sizeof(u32);
    size_t bytes_lagrange = n_vars*n_samps*n_samps * sizeof(u32);

    cudaSetDevice(1);

    // Allocate memory on the device
    cudaMalloc(&d_xs, bytes_xs);
    cudaMalloc(&d_probes, bytes_probes);
    cudaMalloc(&d_probes_2, bytes_probes);
    cudaMalloc(&d_lagrange, bytes_lagrange);

    cudaMemcpy(d_xs, xs, bytes_xs, cudaMemcpyHostToDevice);

    int threadsPerBlock = static_cast<int>(pow(n_samps, n_vars))%256;
    int blocksPerGrid = (probe_len + threadsPerBlock - 1) / threadsPerBlock;

    // Computre all probes
    compute_probes<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_probes, d_probes_2, n_vars, n_samps);

    // Compute the expanded lagrange polynomials in canonical form while GPU busy
    for (int i=0; i<n_vars; i++)
    {
        // TODO: do on GPU
        compute_lagrange_pol(xs, lagrange_polynomials, i, n_vars, n_samps);
    }

    cudaMemcpy(d_lagrange, lagrange_polynomials, bytes_lagrange, cudaMemcpyHostToDevice);


    // Perform multidimensional interpolation
    for (int i=0; i<n_vars; i++)
    {
        get_lagrange_coeffs_nd<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_probes, d_probes_2, d_lagrange, i, n_vars, n_samps);
        std::swap(d_probes, d_probes_2);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(probes, d_probes, bytes_probes, cudaMemcpyDeviceToHost);

    std::vector<double> probe_vec(probe_len);
    for (int i=0; i<probe_len; i++)
    {
        std::cout << "probe: " << probes[i] << " ";
        probe_vec[i] = probes[i];
    }


    // for (int i=0; i<n_vars*n_samps*n_samps; i++)
    // {
    //     if (i%n_samps == 0) {
    //         std::cout << std::endl;
    //     }
    //     if (lagrange_polynomials[i] > PRIME / 2){
    //         std::cout << lagrange_polynomials[i] - long(PRIME) << " ";
    //     } else {
    //         std::cout << lagrange_polynomials[i] << " ";
    //     }

    // }

    std::vector<std::string> vars = {"x", "y", "z"};
    std::string poly = nd_poly_to_string_flat(probe_vec, vars, n_samps);
    std::cout << std::endl << poly << std::endl;

    // Free memory on the device
    cudaFree(d_xs);
    cudaFree(d_probes);
    cudaFree(d_probes_2);
    cudaFree(d_lagrange);

}

// int main()
// {
//     multi_interp(3, 6);
//     return 0;
// }