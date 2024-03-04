#ifndef TENSORLITE_H
#define TENSORLITE_H

#include <vector>

// A multi-dimensional array of data
class Tensor {
public:
    Tensor(const std::vector<int>& shape);
    ~Tensor(); // Free CUDA memory

    // Basic operations
    void allocateMemoryOnDevice();
    void copyDataToDevice(const std::vector<float>& data);
    void copyDataToHost(std::vector<float>& data);
    void add(Tensor &other, Tensor &result); // @TODO: Verify if this belongs here

    // Attributes
    float* device_data; // Pointer to device memory
    std::vector<int> shape; // Shape of the tensor
    int totalSize;
};

class Operation {
public:
    virtual void forward(const Tensor& input, Tensor& output) = 0;
    virtual void backward(const Tensor& input, Tensor& gradInput, const Tensor& gradOutput) = 0;
};

/** A dense layer with a linear transformation */
class DenseLayer : public Operation {
public:
    DenseLayer(int inputSize, int outputSize);
    void forward(const Tensor& input, Tensor& output) override;
    void backward(const Tensor& input, Tensor& gradInput, const Tensor& gradOutput) override;

    int inputSize;
    int outputSize;
    Tensor* weights;
    Tensor* bias;
};

#endif //TENSORLITE_H
