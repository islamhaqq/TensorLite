#include "TensorLite.h"
#include <cuda_runtime.h>
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape) : shape(shape), device_data(nullptr) {
    // Compute total size
    totalSize = 1;
    for (int dim : shape) totalSize *= dim; // Multiply all dimensions together
}

Tensor::~Tensor() {
    cudaFree(device_data);
}

void Tensor::allocateMemoryOnDevice() {
    cudaMalloc(&device_data, totalSize * sizeof(float));
}

void Tensor::copyDataToDevice(const std::vector<float>& data) {
    cudaMemcpy(device_data, data.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::copyDataToHost(std::vector<float>& data) {
    data.resize(totalSize);
    cudaMemcpy(data.data(), device_data, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
}

void Tensor::add(Tensor& other, Tensor& result) {
    if (this->totalSize != other.totalSize || this->totalSize != result.totalSize) {
        throw std::invalid_argument("Tensors must be of the same size");
    }
    // Use the external CUDA function for addition
    vector_add(result.device_data, this->device_data, other.device_data, this->totalSize);
    cudaDeviceSynchronize();
}