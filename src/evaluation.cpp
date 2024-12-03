#include "GPUFR/evaluation.hpp"

#include <string>
#include <sstream>
#include <set>
#include <stack>
#include <vector>

#include "GPUFR/parser.hpp"

int max_stack_depth(std::vector<std::string> &parsed_expression){
	std::stack<u32> S;
	size_t max_depth = 0;
	for(auto token: parsed_expression){
		if(!is_operator(token)){
			S.push(abs(std::stoi(token)));
			max_depth = S.size() > max_depth ? S.size() : max_depth;
		}
		else{
			u32 R = S.top();
			S.pop();
			u32 L = S.top();
			S.pop();

			S.push(evaluate_cpu(token, L, R));
			max_depth = S.size() > max_depth ? S.size() : max_depth;
		}
	}

	return S.top();
}

u32 evaluate_cpu(std::string &op, u32 L, u32 R){
	std::string function_name;

		if(op == "+"){ 
		    return L + R;
		} else
		if(op == "-"){ 
			return L - R;
		} else
		if(op == "*"){ 
			return L * R;
		} else
		if(op == "/"){ 
		    return L / R;
		} else
		if(op == "^"){ 
			u32 res = 1;
			while(R){
			    res *= res;
			    R -= 1;
			}
			return res;
		}
		else{
		    return 0;
		}
}
