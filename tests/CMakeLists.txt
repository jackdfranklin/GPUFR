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

add_executable(vm_test vandermonde_test.cu)
target_link_libraries(vm_test PRIVATE gpufr Catch2::Catch2WithMain ${FLINT_LINK_LIBRARIES})
target_include_directories(vm_test PRIVATE ${FLINT_INCLUDE_DIRS})
set_target_properties(vm_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(ui_test univariate_interpolation_test.cu)
target_link_libraries(ui_test PRIVATE gpufr Catch2::Catch2WithMain ${FLINT_LINK_LIBRARIES})
target_include_directories(ui_test PRIVATE ${FLINT_INCLUDE_DIRS})
set_target_properties(ui_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(parser_test parser_test.cpp)
target_link_libraries(parser_test PRIVATE gpufr Catch2::Catch2WithMain ${FLINT_LINK_LIBRARIES} CUDA::nvrtc)
target_include_directories(parser_test PRIVATE ${FLINT_INCLUDE_DIRS})
set_target_properties(parser_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bb_test bb_gen_test.cpp)
target_link_libraries(bb_test PRIVATE gpufr Catch2::Catch2WithMain ${FLINT_LINK_LIBRARIES} CUDA::nvrtc)
target_include_directories(bb_test PRIVATE ${FLINT_INCLUDE_DIRS})
set_target_properties(bb_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Register tests for ctest/make test to run
add_test(NAME CudaTest COMMAND cuda_test) # Command can be a target
add_test(NAME FiniteFieldTest COMMAND ff_test) 
add_test(NAME VandermondeSolverTest COMMAND vm_test)
add_test(NAME UnivariateInterpolationTest COMMAND ui_test)
add_test(NAME ParserTest COMMAND parser_test)
add_test(NAME BBGenTest COMMAND bb_test)
