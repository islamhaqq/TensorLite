# TensorLite: A Simplified TensorFlow-Inspired Framework

## Overview

TensorLite is a lightweight, educational framework inspired by TensorFlow, one of the most popular and comprehensive libraries for machine learning and deep learning. This project aims to provide a basic understanding of how deep learning frameworks operate under the hood by simulating the essential components of TensorFlow in a significantly simplified manner.

TensorLite allows users to define tensors, perform basic tensor operations (addition, multiplication, matrix multiplication), and execute these operations within a simplistic session-like environment. It is designed for educational purposes to help beginners grasp the core concepts behind computational graphs, tensor operations, and the execution model used in more sophisticated frameworks like TensorFlow.

## Features

- **Tensor Creation**: Users can create tensors, which are the fundamental building blocks of TensorLite, similar to how tensors are used in TensorFlow.
- **Basic Operations**: TensorLite supports basic tensor operations such as addition, multiplication, and matrix multiplication.
- **Simplified Session**: A simplified session mechanism to execute the defined operations, mirroring the execution model of TensorFlow but in a more accessible manner.

## Usage

TensorLite is intended for educational use and to provide insight into the workings of computational graphs and tensor operations. It is not meant for production-level applications or for performance-intensive computations. Users can define their tensors, set up operations, and run these within a session to see the results.

Here's a simple example of how to use TensorLite:

```python
from miniflow import Tensor, add, multiply, matmul, Session

# Define tensors
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Create operations
add_op = add([a, b])
mul_op = multiply([a, b])
matmul_op = matmul([a, b])

# Run session
sess = Session()
print("Addition Result:", sess.run(add_op))
print("Multiplication Result:", sess.run(mul_op))
print("Matrix Multiplication Result:", sess.run(matmul_op))
```

## Limitations

TensorLite is a highly simplified version of TensorFlow and does not support advanced features such as automatic differentiation, GPU acceleration, or a wide range of operations and utilities available in TensorFlow. It is designed purely for educational purposes to help users understand the basic concepts behind tensor operations and computational graphs.

## Contributions

Contributions to TensorLite are welcome! Whether it's extending functionality, improving the documentation, or fixing bugs, your help can make TensorLite a better educational tool for everyone interested in learning about deep learning frameworks.