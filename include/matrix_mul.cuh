#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

// Kernel function decleration

__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width);

extern "C" void simple_test_function() {
    // Just an empty function for testing DLL export
}

#endif //MATRIX_MUL_CUH