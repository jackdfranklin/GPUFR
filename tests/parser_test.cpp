#include "GPUFR/parser.hpp"
#include "GPUFR/types.hpp"

#include <array>
#include <vector>
#include <deque>
#include <stack>
#include <algorithm>
#include <random>
#include <sstream>
#include <filesystem>

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

	SECTION("Conversion from operator to function is correct"){

		const std::string expression = "3+x^2+7*x^3*y^2";
		const std::string ff_expression = "ff_add(ff_add(3, ff_pow(x, 2, p), p), ff_multiply(ff_multiply(7, ff_pow(x, 3, p), p), ff_pow(y, 2, p), p), p)";

		std::vector<std::string> rpn = parse_expression(expression);

		std::string parsed_expr = postfix_to_ff(rpn);

		REQUIRE(parsed_expr == ff_expression);
	}
}
