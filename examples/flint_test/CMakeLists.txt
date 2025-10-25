add_executable(flint_test flint_test.cpp)
target_link_libraries(flint_test PRIVATE ${FLINT_LINK_LIBRARIES})
target_include_directories(flint_test PRIVATE ${FLINT_INCLUDE_DIRS})