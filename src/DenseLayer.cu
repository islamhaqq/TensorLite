#include "TensorLite.h"

// CUDA kernel for forward pass in dense layer
__global__ void denseForwardKernel(float* input, float* weights, float* bias, float* output, int inputSize, int outputSize) {
    // @TODO: CUDA code for matrix multiplication plus bias addition
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
    // Launch the CUDA kernel @TODO: Actual arguments need to be calculated based on block and grid sizes
    denseForwardKernel<<<1, 256>>>(input.device_data, weights->device_data, bias->device_data, output.device_data, inputSize, outputSize);
}

void DenseLayer::backward(const Tensor &input, Tensor &gradInput, const Tensor &gradOutput) {
    // Use the denseBackwardKernel to compute the backward pass and update gradients
    // Launch the CUDA kernel @TODO: Actual arguments need to be calculated based on block and grid sizes
    denseBackwardKernel<<<1, 256>>>(gradOutput.device_data, weights->device_data, gradInput.device_data, inputSize, outputSize);
}
