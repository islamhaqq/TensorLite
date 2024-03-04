cmake_minimum_required(VERSION 3.28)
project(TensorLite LANGUAGES CXX CUDA)

# Set the C++ standard and output directories
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Include directories
include_directories(include)

# Add library
add_library(TensorLite SHARED
        src/vector_add.cu
        src/matrix_mul.cu
        src/TensorLite.cpp
        src/TensorLite.h
)

set_target_properties(TensorLite PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
)

# Set properties for the library
set_target_properties(TensorLite PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "60;61;70;75;80;86"
)

# Include CUDA directories
target_include_directories(TensorLite PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
