#include "GPUFR/parser.hpp"
#include "GPUFR/types.h"

#include <array>
#include <vector>
#include <deque>
#include <stack>
#include <algorithm>
#include <random>
#include <sstream>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <flint/flint.h>
#include <flint/nmod_types.h>
#include <flint/nmod.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

TEST_CASE("Parser Test", "[Parsing][Evaluation]"){

	SECTION("Tokenization should reproduce initial string"){
		const std::string expression = "3+x^2+7*x^3";
		
		std::deque<std::string> tokens = tokenize(expression);

		std::stringstream result;
		for(auto token: tokens){
			result << token;
		}

		REQUIRE(result.str() == expression);

	}

	SECTION("Parsing should reproduce initial string"){

		const std::string expression = "3+x^2+7*x^3";

		std::vector<std::string> rpn = parse_expression(expression);

		std::string rpn_string;
		for(auto s: rpn){
			rpn_string += s;
		}
		INFO(rpn_string);
		std::string parsed_expr = postfix_to_infix(rpn);

		REQUIRE(parsed_expr == expression);
	}

	const std::vector<std::string> vars = {"x"};
	const std::string expression = "3 + x^2 + 7*x^3";
	auto black_box = [](u32 x, nmod_t mod){
		u32 result = 3;
		result = nmod_add(result, nmod_pow_ui(x, 2, mod), mod);
		u32 temp = nmod_mul(7, nmod_pow_ui(x, 3, mod), mod);
		result = nmod_add(result, temp, mod);
		return result;
	};

	//std::string cuda_code = cuda_from_expression(expression, vars);

	//nvrtcProgram program;
	//nvrtcCreateProgram(&program, cuda_code.data(), 0, NULL, NULL);




}
