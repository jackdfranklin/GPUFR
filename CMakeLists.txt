cmake_minimum_required(VERSION 3.20)

project(GPUFR VERSION 0.1
			  DESCRIPTION "Black box function reconstruction on GPUs"
			  LANGUAGES CXX CUDA)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

		include(CTest)

		add_subdirectory(tests)
endif()

include(FetchContent)

add_subdirectory(src)


