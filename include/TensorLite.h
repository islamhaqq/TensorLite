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

#endif //TENSORLITE_H
