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
    T* ptr_bottom;

    public:

	__device__ stack(u32* stack_allocation, int thread_id, int max_stack)
    {
        // ptr_bottom = &data;
        ptr_bottom = stack_allocation + thread_id*max_stack;
        ptr_top = ptr_bottom-1;
    }

    __device__ void push(T val)
    {
        ptr_top += 1;
        *ptr_top = val;
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

    __host__ cu_string(const std::string &r_val)
    {
        int len = size_val < r_val.size()? size_val : r_val.size();
        for (int i=0; i<len; i++)
        {
            data[i] = r_val[i];
        }
        data[len] = '\0';
    }

    template<int r_size>
    __host__ cu_string(const char (&r_val)[r_size])
    {
        int len = size_val < r_size? size_val : r_size;
        for (int i=0; i<len; i++)
        {
            data[i] = r_val[i];
        }
        data[len] = '\0';
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
            if (data[i] == '\0')
                return true;
        }
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

    __host__ __device__ inline const char* c_str() const
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
        data[len] = '\0';
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
__host__ __device__ u32 strtou(const cu_string<n>& in)
{
    u32 result = 0;
    for (int i=0; i<n; i++)
    {
        if (in[i] == '\0')
        {
            return result;
        }
        result *= 10;
        result += (in[i] - '0');
    }
} 

__device__ u32 detokenize(u32* stack_allocation, const cu_string<STRING_LEN>* tokens, const cu_string<STRING_LEN>* var_labels, int max_stack, int thread_id, int token_len, int n_vars, u32 *vars, u32 prime);

__device__ u32 to_u32(cu_string<STRING_LEN> &token, u32 *vars, const cu_string<STRING_LEN>* var_labels, int n_vars);

__device__ bool is_operator(const cu_string<STRING_LEN> &token);

__device__ u32 operator_to_function(const cu_string<STRING_LEN> &op, u32 L, u32 R, u32 prime);

__host__ cu_string<STRING_LEN>* to_cu_string(const std::vector<std::string> &tokens);
