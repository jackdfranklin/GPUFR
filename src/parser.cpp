#include "GPUFR/parser.hpp"

#include <string>
#include <sstream>
#include <set>
#include <stack>
#include <vector>

std::string gen_kernel(const std::vector<std::string> &vars){

	std::string kernel_string = "						\n\
	extern \"C\" __global__ void evaluate(u32 n_values, u32 *values, u32 *result, u32 p){	\n\
		size_t idx = threadIdx.x + blockIdx.x * blockDim.x; 		\n";

	size_t n_vars = vars.size();

	std::string var_string;

	for(size_t vidx = 0; vidx < n_vars; vidx++){
		var_string += "values[idx + " + std::to_string(vidx) + "*n_values], ";
	}

	kernel_string += "result[idx] = black_box("+var_string+"p);\n";
	kernel_string += "}";

	return kernel_string;

}

std::string cuda_from_expression(const std::string &expression, const std::vector<std::string> &vars){

	std::string var_string;
	for(auto var: vars){
		var_string += "u32 " + var + ", ";
	}

	std::string program_string = "#include \"GPUFR/ff_math.cuh\"\n";
	program_string += "#include \"GPUFR/types.hpp\"\n";
	program_string += "__device__ u32 black_box(\n";
	program_string += var_string;
	program_string += "u32 p){\n";

	program_string += "return ";
	program_string += postfix_to_ff(parse_expression(expression)) + ";\n";
	program_string += "}\n";

	program_string += gen_kernel(vars);

	return program_string;
}

std::vector<std::string> parse_expression( const std::string &expression ){

	std::deque<std::string> tokens = tokenize(expression);

	std::vector<std::string> rpn_expression;
	std::stack<std::string> operator_stack;

	for(auto token: tokens){
		if(is_operator(token)){
			if(operator_stack.empty() || operator_stack.top() == "("){
				operator_stack.push(token);
			} else
			if(precedence(token) > precedence(operator_stack.top())){
				operator_stack.push(token);
			} else
			if(precedence(token) == precedence(operator_stack.top()) && right_assoc(token)){
				operator_stack.push(token);
			} 
			else {
				while(!operator_stack.empty() && should_pop(token, operator_stack.top())){
					rpn_expression.push_back(operator_stack.top());
					operator_stack.pop();
				}
				operator_stack.push(token);
			}
		} else
		if(token == "("){
			operator_stack.push(token);
		} else
		if(token == ")"){
			while(!operator_stack.empty() && operator_stack.top() != "("){
				rpn_expression.push_back(operator_stack.top());
				operator_stack.pop();
			}

		}
		else {
			rpn_expression.push_back(token);
		}
	}

	while(!operator_stack.empty()){
		rpn_expression.push_back(operator_stack.top());
		operator_stack.pop();
	}

	return rpn_expression;
}

std::deque<std::string> tokenize(const std::string &expression){
	std::deque<std::string> tokens;

	std::string tmp;

	std::string operator_str = "+-*/^()";
	std::set<char> operators(operator_str.begin(), operator_str.end());

	for(auto &c: expression ){
		if( operators.count(c) > 0 ){

			tokens.push_back(tmp);
			tmp.clear();

			tmp.push_back(c);
			tokens.push_back(tmp);
			tmp.clear();
		}	

		if( isdigit(c) || isalpha(c) ){
			tmp.push_back(c);
		}
	}

	tokens.push_back(tmp);

	return tokens;
}

bool is_operator(const std::string &token){
	if(token == "+" || token == "-" || token == "*" || token == "/" || token == "^"){
		return true;
	}
	else {
		return false;
	}
}

int precedence(const std::string &token){
	if(token == "^"){
		return 3;
	} else
	if(token == "*" || token == "/"){
		return 2;
	} else
	if(token == "+" || token == "-"){
		return 1;
	}
	else{
		return 0;
	}
}

bool right_assoc(const std::string &token){
	return (token == "^");
}

bool should_pop(const std::string &token, const std::string &top_op){
	return (precedence(token) < precedence(top_op)) || (precedence(token) == precedence(top_op) && !right_assoc(token));
}

std::string postfix_to_ff(const std::vector<std::string> &rpn){
	std::stack<std::string> S;
	for(auto token: rpn){
		if(!is_operator(token)){
			S.push(token);
		}
		else{
			std::string R = S.top();
			S.pop();
			std::string L = S.top();
			S.pop();

			S.push(operator_to_function(token, L, R));
		}
	}

	return S.top();
	
}

std::string operator_to_function(const std::string &op, const std::string &L, const std::string &R){
	std::string function_name;

		if(op == "+"){ 
			function_name = "ff_add";
		} else
		if(op == "-"){ 
			function_name = "ff_subtract";
		} else
		if(op == "*"){ 
			function_name = "ff_multiply";
		} else
		if(op == "/"){ 
			function_name = "ff_divide";
		} else
		if(op == "^"){ 
			function_name = "ff_pow";
		}
		else{
			function_name = "invalid_op_"+op;
		}

	return function_name+"("+L+", "+R+", p)";
}

std::string postfix_to_infix(const std::vector<std::string> &rpn){
	std::stack<std::string> S;
	for(auto token: rpn){
		if(!is_operator(token)){
			S.push(token);
		}
		else{
			std::string R = S.top();
			S.pop();
			std::string L = S.top();
			S.pop();

			S.push(L+token+R);
		}
	}

	return S.top();
}
