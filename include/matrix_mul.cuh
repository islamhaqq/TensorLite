#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

extern "C" void matrixMul(const float *h_A, const float *h_B, float *h_C, int width);

#endif //MATRIX_MUL_CUH