include(FetchContent)
# Testing library 
FetchContent_Declare(
	catch
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG v3.6.0)
FetchContent_MakeAvailable(catch)
# Adds Catch2::Catch2

add_executable(ff_test ff_test.cpp)
target_link_libraries(ff_test PRIVATE Catch2::Catch2WithMain)

add_executable(cuda_test cuda_test.cu)
target_link_libraries(cuda_test PRIVATE Catch2::Catch2WithMain)

# Register tests for ctest/make test to run
add_test(NAME FiniteFieldTest COMMAND ff_test) # Command can be a target
add_test(NAME CudaTest COMMAND cuda_test) 
