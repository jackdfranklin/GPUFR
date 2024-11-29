#pragma once

#include "GPUFR/types.hpp"
#include "GPUFR/ff_math.cuh"

#include <vector>
#include <string>
#include <stack>

#define STRING_LEN 10 // 10 chars to write max unsigned in denary

template<typename T>
class stack
{
    private:
    T* ptr_top;
    // T* ptr_bottom;
    T ptr_bottom[100];

    public:

	__device__ stack()
    {
        // ptr_bottom = &data;
        ptr_top = ptr_bottom;
    }

    __device__ void push(T val)
    {
        *ptr_top = val;
        ptr_top += 1;
    }

    __device__ void  pop()
    {
        ptr_top -= 1;
    }

    __device__ T top()
    {
        return *ptr_top;
    }
};

template<int size_val>
class cu_string
{
    private:
    char data[size_val];

    public:
    __host__ __device__ cu_string()
    {

    }

    __host__ __device__ inline char& operator[](int index)
    {
        return data[index];
    }

    __host__ __device__ inline char operator[](int index) const
    {
        return data[index];
    }

    __host__ __device__ inline bool operator==(cu_string r_val)
    {
        for (int i=0; i<size_val; i++)
        {
            if (data[i] != r_val[i])
                return false;
        }

        return true;
    }

    __host__ __device__ inline int size()
    {
        return size_val;
    }

    __host__ __device__ inline cu_string<size_val> substr(int start)
    {
        cu_string<size_val> result;
        for (int i=0; i<size_val-start; i++)
        {
            result[i] = data[i+start];
        }

        return result;
    }

    __host__ __device__ inline char* c_str()
    {
        return data;
    }

    __host__ inline void operator=(const std::string &r_val)
    {
        int len = size_val < r_val.size()? size_val : r_val.size();
        for (int i=0; i<len; i++)
        {
           data[i] = r_val[i];
        }
        data[size_val - 1] = '\0';
    }
};

template<int m, int n>
__host__ __device__ inline bool operator==(const cu_string<m>& l_val, const char (&r_val)[n])
{
    int len = m < n? m : n;
    for (int i=0; i<len; i++)
    {
        if (l_val[i] != r_val[i])
            return false;
    }

    return true;
}

template<int n>
__device__ u32 strtou(const cu_string<n>& in)
{
    u32 result = 0;
    u32 base = 1;
    for (int i=0; i<n; i++)
    {
        result += base*(in[n-i] - '0');
        base *= 10;
    }

    return result;
} 

__device__ u32 detokenize(const cu_string<STRING_LEN>* tokens, const cu_string<STRING_LEN>* var_labels, int exp_len, int n_vars, u32 *vars, u32 prime);

__device__ u32 to_u32(cu_string<STRING_LEN> &token, u32 *vars, const cu_string<STRING_LEN>* var_labels, int n_vars);

__device__ bool is_operator(const cu_string<STRING_LEN> &token);

__device__ u32 operator_to_function(const cu_string<STRING_LEN> &op, u32 L, u32 R, u32 prime);

__host__ cu_string<STRING_LEN>* to_cu_string(const std::vector<std::string> &tokens);
