cmake_minimum_required(VERSION 3.26)

project(GPUFR VERSION 0.1
			  DESCRIPTION "Black box function reconstruction on GPUs"
			  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

# Release flags (add optimizations)
set(CMAKE_CXX_FLAGS_RELEASE " -g3 -G")
set(CMAKE_CUDA_FLAGS_RELEASE " -g3 -G")
find_package(CUDAToolkit REQUIRED)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

		if (CMAKE_BUILD_TYPE STREQUAL "Debug")
			include(CTest)

			add_subdirectory(tests)
		else()
			add_subdirectory(examples)
			add_subdirectory(tests)
		endif()
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include(FetchContent)

add_subdirectory(src)


