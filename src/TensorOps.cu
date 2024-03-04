#include "TensorLite.h"

// Kernel for element-wise addition
__global__ void addKernel(float *a, float *b, float *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the index of the current thread
    if (idx < N) result[idx] = a[idx] + b[idx]; // Perform the addition
}

// Function to call the kernel
void Tensor::add(Tensor &other, Tensor &result) {
    int N = this->totalSize; // Assuming all tensors are the same size
    int blockSize = 256; // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // Number of blocks needed to cover the entire array
    addKernel<<<numBlocks, blockSize>>>(this->device_data, other.device_data, result.device_data, N); // Launch the kernel
    cudaDeviceSynchronize(); // Wait for the GPU to finish
}