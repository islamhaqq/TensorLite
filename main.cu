#include <iostream>
#include <iomanip> // For setting precision when printing floats
#include <cuda_runtime.h> // Necessary for CUDA runtime APIs
#include "matrix_mul.cuh"

void initializeMatrix(float *mat, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i * width + j] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
        }
    }
}

void printMatrix(const float *mat, int width, int height) {
    for (int i = 0; i < std::min(10, height); i++) {
        for (int j = 0; j < std::min(10, width); j++) {
            std::cout << std::fixed << std::setprecision(2) << mat[i * width + j] << " ";
        }
        std::cout << (width > 10 ? "... (truncated)" : "") << std::endl;
    }
    std::cout << (height > 10 ? "... (truncated rows)" : "") << std::endl;
}

int main()
{
    int width = 1024; // Define matrix size (width * width for simplicity)
    size_t size = width * width * sizeof(float); // Calculate the size of the matrix in bytes

    float *h_M, *h_N, *h_P; // Host matrices
    float *d_M, *d_N, *d_P; // Device matrices

    // Allocate memory on the host
    h_M = (float *)malloc(size);
    h_N = (float *)malloc(size);
    h_P = (float *)malloc(size);

    // Initialize matrices with random values
    initializeMatrix(h_M, width, width);
    initializeMatrix(h_N, width, width);

    // Allocate memory on the device
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    // Copy matrices from the host to the device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Define block size and grid si ze
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_P, width);

    // Copy the result back to the host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Print a portion of the resulting matrix
    std::cout << "Resulting matrix (portion):" << std::endl;
    printMatrix(h_P, width, width);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
