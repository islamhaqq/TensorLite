#include "TensorLite.h"

// CUDA kernel for forward pass in dense layer
__global__ void denseForwardKernel(float* input, float* weights, float* bias, float* output, int inputSize, int outputSize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the index of the current thread
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < outputSize && col < inputSize) { // Ensure the current thread is within the array bounds
        float sum = 0;
        for (int i = 0; i < inputSize; ++i) { // Perform the matrix  multiplication
            sum += input[i] * weights[row * inputSize + i];
        }
        output[row] = sum + bias[row]; // Add the bias
    }
}

__global__ void denseBackwardKernel(float* gradOutput, float* weights, float* gradInput, int inputSize, int outputSize) {
    // @TODO: CUDA code for backpropagation through dense layer
}

DenseLayer::DenseLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize) {
    // Allocate memory for weights and biases, and initialize them
    weights = new Tensor({inputSize, outputSize});
    bias = new Tensor({outputSize});
    weights->allocateMemoryOnDevice();
    bias->allocateMemoryOnDevice();
}

void DenseLayer::forward(const Tensor &input, Tensor &output) {
    // Use the denseForwardKernel to compute the forward pass
    dim3 blockSize(16, 16);
    dim3 gridSize((input.shape[1] + blockSize.x - 1) / blockSize.x, (weights->shape[0] + blockSize.y - 1) / blockSize.y);
    denseForwardKernel<<<gridSize, blockSize>>>(input.device_data, weights->device_data, bias->device_data, output.device_data, inputSize, outputSize);
}

void DenseLayer::backward(const Tensor &input, Tensor &gradInput, const Tensor &gradOutput) {
    // Use the denseBackwardKernel to compute the backward pass and update gradients
    dim3 blockSize(16, 16);
    dim3 gridSize((input.shape[1] + blockSize.x - 1) / blockSize.x, (weights->shape[0] + blockSize.y - 1) / blockSize.y);
    denseBackwardKernel<<<blockSize, gridSize>>>(gradOutput.device_data, weights->device_data, gradInput.device_data, inputSize, outputSize);
}
