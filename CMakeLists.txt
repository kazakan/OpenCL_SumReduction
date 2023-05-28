
cmake_minimum_required (VERSION 3.8)

project ("OpenCL_Sum_Reduction_Example" LANGUAGES CXX)

add_executable (run "main.cpp" "main.h")

# OPENCL

FIND_PACKAGE(OpenCL REQUIRED)

INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS}) 

LINK_DIRECTORIES(${OpenCL_LIBRARY})

target_link_libraries(run OpenCL::OpenCL)

# prevent std segfault
target_link_options(run PRIVATE -static-libgcc -static-libstdc++ -static)

