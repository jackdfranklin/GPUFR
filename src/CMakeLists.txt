file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${GPUFR_SOURCE_DIR}/include/GPUFR/*.hpp" "${GPUFR_SOURCE_DIR}/include/GPUFR/*.cuh")

add_library(gpufr ff_math.cu vandermonde_solver.cu lagrange_solver.cu ntt.cu ${HEADER_LIST}) 
target_include_directories(gpufr PUBLIC ../include )
set_target_properties(gpufr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
