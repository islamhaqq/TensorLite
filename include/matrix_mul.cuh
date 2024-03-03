#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

// Kernel function decleration

__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width);

#endif //MATRIX_MUL_CUH