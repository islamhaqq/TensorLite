cmake_minimum_required(VERSION 3.28)
project(TensorLite LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 14)

# Add library
add_library(TensorLite SHARED
        src/vector_add.cu
        src/matrix_mul.cu
        src/TensorLite.cpp
        include/TensorLite.h
)

# Set properties for the library
set_target_properties(TensorLite PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "60;61;70;75;80;86"
)

# Include directories
target_include_directories(TensorLite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Log CUDA status
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA Compiler not found")
endif()

# Include CUDA Toolkit directories (if found)
if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    message(STATUS "CUDA Toolkit Include Directories: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    target_include_directories(TensorLite PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
    message(WARNING "CUDA Toolkit include directories not found.")
endif()
