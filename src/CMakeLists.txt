
add_executable (run "main.cpp" "main.h")
set(OCL_KERNELS "sum.cl")

# OPENCL

INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${LIB_INCLUDE}) 

LINK_DIRECTORIES(${OpenCL_LIBRARY})

target_include_directories(run PRIVATE ${LIB_INCLUDE})
target_link_libraries(run OpenCL::OpenCL)

# prevent std segfault
target_link_options(run PRIVATE -static-libgcc -static-libstdc++ -static)

# copy kernel
foreach(KERNEL_PATH ${OCL_KERNELS})
    file(COPY ${KERNEL_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
