#include "GPUFR/parser.hpp"

#include <string>
#include <sstream>
#include <set>
#include <stack>
#include <vector>

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
