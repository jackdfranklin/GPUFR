cmake_minimum_required(VERSION 3.20)

project(GPUFR VERSION 0.1
			  DESCRIPTION "Black box function reconstruction on GPUs"
			  LANGUAGES CXX CUDA)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

		include(CTest)

		add_subdirectory(tests)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include(FetchContent)

add_subdirectory(src)


