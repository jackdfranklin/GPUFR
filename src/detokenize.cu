#include "GPUFR/detokenize.cuh"

// subdivide token vector into simpler vectors and pass them to kernels
// impliment a simple stack on the kernel cache

// Parsing on the gpu is difficult as strings are not supported, vecotors are easy enough to impliment so are stacks
// Parsing on the cpu then calling many kernels likley incurrs more overhead but will be more streight forward.
// In both cases, a large amount of memory will be reqiured to store the intermediate expressions. Per thread this could be as large as (n_samps*n_vars)^2,
// i.e. the number of elements in the coefficient matrix. Or perhapse not??? Reconstructing a single coefficient given N starting integrals, there could be as many
// as N intermediate expressions, although probably not.

// Compute all probes before allocating the memory for the reconstruction to avoid running out

__device__ u32 detokenize(u32* stack_allocation, const cu_string<STRING_LEN>* tokens, const cu_string<STRING_LEN>* var_labels, int max_stack, int thread_id, int token_len, int n_vars, u32 *vars, u32 prime)
{
	stack<u32> US(stack_allocation, thread_id, max_stack);
	for(int i=0; i<token_len; i++){
		cu_string token = tokens[i];
		if(!is_operator(token)){
            US.push(to_u32(token, vars, var_labels, n_vars));
		}
		else{
			u32 R = US.top();
			US.pop();
			u32 L = US.top();
			US.pop();

			US.push(operator_to_function(token, L, R, prime));
		}
	}

	return US.top();
}

__device__ u32 to_u32(cu_string<STRING_LEN> &token, u32 *vars, const cu_string<STRING_LEN>* var_labels, int n_vars)
{
    int count = 0;
    for (int i=0; i<n_vars; i++)
    {
		cu_string v = var_labels[i];

        if (token == v)
        {
            return vars[count];
        }
        count += 1;
    }
    return strtou(token);
}

__device__ bool is_operator(const cu_string<STRING_LEN> &token)
{
	if(token == "+" || token == "-" || token == "*" || token == "/" || token == "^"){
		return true;
	}
	else {
		return false;
	}
}

__device__ u32 operator_to_function(const cu_string<STRING_LEN> &op, u32 L, u32 R, u32 prime){
	u32 result;

		if(op == "+"){ 
			result = ff_add(L, R, prime);
		} else
		if(op == "-"){ 
			result = ff_subtract(L, R, prime);
		} else
		if(op == "*"){ 
			result = ff_multiply(L, R, prime);
		} else
		if(op == "/"){ 
			result = ff_divide(L, R, prime);
		} else
		if(op == "^"){ 
			result = ff_pow(L, R, prime);
		}

	return result;
}

__host__ cu_string<STRING_LEN>* to_cu_string(const std::vector<std::string> &tokens)
{
	cu_string<STRING_LEN>* result = new cu_string<STRING_LEN>[tokens.size()];
	for (int i=0; i<tokens.size(); i++)
	{
		result[i] = tokens[i];
	}

	return result;
}
