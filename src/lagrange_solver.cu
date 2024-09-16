#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip> 

#include "cuda_safe_call.hpp"
#include "FiniteField.cuh"

#define MAX_VARS 5 // The maximum number of variable to reconstruct over
#define MAX_EXPONENT 20 // The maximum exponent in the polynomeal
#define UNSIGNED_TYPE unsigned
#define PRIME 105097565 // Must be less than (max unsigend) / 2

__device__ FiniteField<UNSIGNED_TYPE> fun(FiniteField<UNSIGNED_TYPE> *vars)
{
    FiniteField<UNSIGNED_TYPE> result;
    result.set_prime(PRIME);
    result = 1;
    return result;
}

__global__ void compute_probes(const FiniteField<UNSIGNED_TYPE> *xs, FiniteField<UNSIGNED_TYPE> *probes, FiniteField<UNSIGNED_TYPE> *probes_2, int n_vars, int n_samps) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    FiniteField<UNSIGNED_TYPE> test_params[MAX_VARS];
    probes[i].set_prime(PRIME);
    probes_2[i].set_prime(PRIME);

    for (int j=0; j<n_vars; j++)
    {
        float ifloat = i;
        int dimension_index = static_cast<int>(floorf(ifloat/pow(n_samps, j))) % n_samps;//floorf((pow(n_samps, j)));
        test_params[j].set_prime(PRIME);
        test_params[j] = xs[j*n_samps+dimension_index];
    }
        
    probes[i] = fun(test_params);
    probes_2[i] = 0;
}


__device__ void atomic_add(FiniteField<UNSIGNED_TYPE> *l_val, FiniteField<UNSIGNED_TYPE> r_val)
{
    UNSIGNED_TYPE assumed, old;
    FiniteField<UNSIGNED_TYPE> value;
    value.set_prime(PRIME);

    old = (*l_val).value();
    do
    {
        value = *l_val;
        assumed = old;
        old = atomicCAS(&((*l_val)._value), assumed, (value + r_val).value());
    } while (assumed != old);
}

__device__ FiniteField<UNSIGNED_TYPE> compute_denom_nd(int current_index, const FiniteField<UNSIGNED_TYPE> *xs, int dim, int n_vars, int n_samps, int idx)
{
    int flat_current_index = dim*n_samps + static_cast<int>(floorf(idx/pow(n_samps, dim)))%n_samps;

    FiniteField<UNSIGNED_TYPE> denom;
    denom.set_prime(PRIME);
    denom = 1;

    for (int i=0; i<n_samps; i++)
    {
        int flat_index = dim*n_samps + i;
        if (flat_index != flat_current_index)
        {
            denom = denom*(xs[flat_current_index] - xs[flat_index]);
        }
    }

    return denom;
}

__global__ void get_lagrange_coeffs_nd(const FiniteField<UNSIGNED_TYPE> *xs, FiniteField<UNSIGNED_TYPE> *ys, FiniteField<UNSIGNED_TYPE> *out, FiniteField<UNSIGNED_TYPE> *lagrange, int dim, int n_vars, int n_samps)
{
    // Computes the coefficients to the Lagrange polynomials and writes them to ys

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    FiniteField<UNSIGNED_TYPE> denom;
    denom.set_prime(PRIME);
    denom = compute_denom_nd(idx, xs, dim, n_vars, n_samps, idx);

    FiniteField<UNSIGNED_TYPE> coefficient = ys[idx] / denom;
    ys[idx] = 0;

    int index_step_ys = pow(n_samps, dim);
    int power = static_cast<int>(floorf(pow(n_samps, dim+1)));
    int start_index_ys = floorf(idx / power)*power + idx%index_step_ys;

    int index_xs = dim*n_samps + static_cast<int>(floorf(idx/pow(n_samps, dim)))%n_samps;

    for (int i=0; i<n_samps; i++)
    {
        int flat_index_ys = start_index_ys+i*index_step_ys;
        int flat_index_lagrange = index_xs*n_samps + i;
        // printf("idx: %i i: %i probe index: %i start index: %i to add: %i denom: %i lagind: %i val: %i \n", idx, i, flat_index_ys, power, coefficient.value(), denom.value(), flat_index_lagrange, (lagrange[flat_index_lagrange]*coefficient).value());
        atomic_add(&out[flat_index_ys], lagrange[flat_index_lagrange]*coefficient);
    }
}


void convolve_cpp(const FiniteField<UNSIGNED_TYPE> *kernel, const FiniteField<UNSIGNED_TYPE> *signal, FiniteField<UNSIGNED_TYPE> *out, int kernel_size, int signal_size)
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
                out[i] += kernel[kernel_size - 1 - j] * signal[i+j-pad_size];
            }
        }
    }
}

void compute_lagrange_pol(const FiniteField<UNSIGNED_TYPE> *xs, FiniteField<UNSIGNED_TYPE>* lagrange, int dim, int n_vars, int n_samps)
{
    FiniteField<UNSIGNED_TYPE> root_arr[2];
    FiniteField<UNSIGNED_TYPE> tmp[MAX_VARS];
    FiniteField<UNSIGNED_TYPE> tmp2[MAX_VARS];

    root_arr[0].set_prime(PRIME);
    root_arr[1].set_prime(PRIME);

    for (int i=0; i<n_samps; i++)
    {
        tmp[i].set_prime(PRIME);
        tmp2[i].set_prime(PRIME);
    }

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
                root_arr[0] = 0-xs[x_index];
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

            lagrange[lagrange_index].set_prime(PRIME);
            lagrange[lagrange_index] = tmp[j];
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
    FiniteField<UNSIGNED_TYPE> lagrange_polynomials[n_vars*n_samps*n_samps];

    FiniteField<UNSIGNED_TYPE> probes[probe_len];
    FiniteField<UNSIGNED_TYPE> xs[n_vars*n_samps];
    for (int i=0; i<n_vars; i++)
    {
        for (int j=0; j<n_samps; j++)
        {
            int flat_index = i*n_samps + j;
            xs[flat_index].set_prime(PRIME);
            xs[flat_index] = (j+1);
            // printf("xs: %i \n", (j+1));
        }
    }

    FiniteField<UNSIGNED_TYPE> *d_xs, *d_probes, *d_probes_2, *d_lagrange;

    // Size in bytes for each vector
    size_t bytes_xs = n_vars*n_samps * sizeof(FiniteField<UNSIGNED_TYPE>);
    size_t bytes_probes = probe_len * sizeof(FiniteField<UNSIGNED_TYPE>);
    size_t bytes_lagrange = n_vars*n_samps*n_samps * sizeof(FiniteField<UNSIGNED_TYPE>);

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
        compute_lagrange_pol(xs, lagrange_polynomials, i, n_vars, n_samps);
    }

    cudaMemcpy(d_lagrange, lagrange_polynomials, bytes_lagrange, cudaMemcpyHostToDevice);


    // Perform multidimensional interpolation
    for (int i=0; i<1; i++)
    {
        get_lagrange_coeffs_nd<<<blocksPerGrid, threadsPerBlock>>>(d_xs, d_probes, d_probes_2, d_lagrange, i, n_vars, n_samps);
        std::swap(d_probes, d_probes_2);
    }

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    cudaMemcpy(probes, d_probes, bytes_probes, cudaMemcpyDeviceToHost);

    std::vector<double> probe_vec(probe_len);
    for (int i=0; i<probe_len; i++)
    {
        std::cout << probes[i].value() << " ";
        if (probes[i].value() > 105097565)
        {
            
        }
        probe_vec[i] = probes[i].value();
    }


    // for (int i=0; i<n_vars*n_samps*n_samps; i++)
    // {
    //     if (i%n_samps == 0) {
    //         std::cout << std::endl;
    //     }
    //     if ((lagrange_polynomials[i]).value() > PRIME / 2){
    //         std::cout << (lagrange_polynomials[i]).value() - long(PRIME) << " ";
    //     } else {
    //         std::cout << (lagrange_polynomials[i]).value() << " ";
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

int main()
{
    multi_interp(3, 6);
    return 0;
}