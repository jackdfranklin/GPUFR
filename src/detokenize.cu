#include "GPUFR/detokenize.cuh"

// subdivide token vector into simpler vectors and pass them to kernels
// impliment a simple stack on the kernel cache

__device__ u32 detokenize(const std::vector<std::string> &tokens, const std::vector<std::string> &var_labels, u32 *vars, u32 prime)
{
	stack<u32> US;
	for(auto token: tokens){
		if(!is_operator(token)){
            US.push(to_u32(token, vars, var_labels));
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

__device__ u32 to_u32(std::string &token, u32 *vars, const std::vector<std::string> &var_labels)
{
    int count = 0;
    for (auto v: var_labels)
    {
        if (token == v)
        {
            int index = std::stoul(token.substr(1));
            return vars[count];
        }
        count += 1;
    }
    
    return std::stoul(token);
}

__device__ bool is_operator(const std::string &token)
{
	if(token == "+" || token == "-" || token == "*" || token == "/" || token == "^"){
		return true;
	}
	else {
		return false;
	}
}

__device__ u32 operator_to_function(const std::string &op, u32 L, u32 R, u32 prime){
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