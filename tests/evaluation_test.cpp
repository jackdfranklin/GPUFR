#include <array>
#include <vector>
#include <deque>
#include <stack>
#include <algorithm>
#include <random>
#include <sstream>
#include <filesystem>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <flint/flint.h>
#include <flint/nmod_types.h>
#include <flint/nmod.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

#include "GPUFR/parser.hpp"
#include "GPUFR/evaluation.hpp"
#include "GPUFR/types.hpp"
#include "GPUFR/nvrtc_helper.hpp"


TEST_CASE("Evaluation Test", "[Parsing][Evaluation]"){

	SECTION("Max Stack Depth should be reproduceable"){
		const std::string expression = "3+4^2+7*5^3";

		std::vector<std::string> rpn = parse_expression(expression);

		REQUIRE(max_stack_depth(rpn) == 11);	

	}

}
