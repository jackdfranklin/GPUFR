# Add Catch2::Catch2
include(FetchContent)
FetchContent_Declare(
	catch
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG v3.6.0)
FetchContent_MakeAvailable(catch)

add_executable(cuda_test cuda_test.cu)
target_link_libraries(cuda_test PRIVATE Catch2::Catch2WithMain)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLINT REQUIRED flint)

add_executable(ff_test ff_test.cu)
target_link_libraries(ff_test PRIVATE gpufr Catch2::Catch2WithMain ${FLINT_LINK_LIBRARIES})
target_include_directories(ff_test PRIVATE ${FLINT_INCLUDE_DIRS})
set_target_properties(ff_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Register tests for ctest/make test to run
add_test(NAME CudaTest COMMAND cuda_test) # Command can be a target
add_test(NAME FiniteFieldTest COMMAND ff_test) 
