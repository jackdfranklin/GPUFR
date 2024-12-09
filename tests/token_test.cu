#include "GPUFR/parser.hpp"
#include "GPUFR/detokenize.cuh"

#include <string>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("String_Initialise")
{
    std::string std_str = "123456789";
    cu_type::string<STRING_LEN> cu_str = std_str;
    cu_type::string<10> cu_str_1 = "123";

    printf("std: %s \n", std_str.c_str());
    printf("cu: %s \n", cu_str.c_str());

    printf("==: %i \n", cu_str==cu_str_1);
}

TEST_CASE("to_u32")
{
    cu_type::string<10> cu_str_1 = "123";

    REQUIRE(strtou(cu_str_1) == 123);
}