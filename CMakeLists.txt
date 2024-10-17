cmake_minimum_required(VERSION 3.20)

project(wukong VERSION 0.1 LANGUAGES CXX CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

set(FILES
    cuda.cu
    common.h
)
add_executable(wukong ${FILES})
target_link_libraries(wukong -lcublasLt)

option(CF_TEST "Enable tests" off)
if(CF_TEST)
    enable_testing()
    find_package(GTest)
    add_executable(test_cuda test.cu ${FILES})
    target_link_libraries(test_cuda GTest::gtest_main -lcublasLt)
    gtest_discover_tests(test_cuda)
endif()

