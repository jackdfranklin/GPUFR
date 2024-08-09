file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${GPUFR_SOURCE_DIR}/include/GPUFR/*.hpp" "${GPUFR_SOURCE_DIR}/include/GPUFR/*.cuh")

add_library(gpufr ff_math.cu ${HEADER_LIST})
set_target_properties(gpufr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(gpufr PUBLIC ../include)
