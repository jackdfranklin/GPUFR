file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${GPUFR_SOURCE_DIR}/include/GPUFR/*.hpp" "${GPUFR_SOURCE_DIR}/include/GPUFR/*.cuh")

add_library(gpufr ff_math.cu vandermonde_solver.cu lagrange_solver.cu ntt.cu crt.cu parser.cpp bb_gen.cpp ${HEADER_LIST}) 

target_include_directories(gpufr PUBLIC ../include )
target_link_libraries(gpufr PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::nvJitLink)
set_target_properties(gpufr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpufr PRIVATE HEADER_SRC="${GPUFR_SOURCE_DIR}/include/" FF_MATH_OBJ="$<TARGET_OBJECTS:ff_math>")
