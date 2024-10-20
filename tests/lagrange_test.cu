#include "GPUFR/lagrange_solver.cuh"

#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <flint/flint.h>
#include <flint/nmod_types.h>
#include <flint/nmod.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

TEST_CASE("Lagrange Interpolation"){
    printf("catch \n");
    multi_interp(1, 1); // Dont necessarily need to computea all probes, could just copy a probe to padd out to a power of 2
    REQUIRE(true);
}