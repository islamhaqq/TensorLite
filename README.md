# TensorLite: A Lightweight TensorFlow-Inspired Framework

## Overview

TensorLite is a lightweight framework, inspired by TensorFlow's core principles, designed for minimalistic and rapid computational needs. This streamlined version provides the essentials for tensor operations and basic computational graph functionalities, making it suitable for small-scale projects and for those seeking a more direct and less resource-intensive approach to tensor computations.

## Features

- **Tensor Operations**: Users can perform essential tensor operations including addition, multiplication, and matrix multiplication, tailored for lightweight computation.
- **Compact Session Execution**: Implements a straightforward session-based execution model, facilitating the running of operations with minimal overhead.
- **Minimalist Design**: Focuses on core functionalities, ensuring a small footprint and quick setup, ideal for lightweight applications and educational purposes.

## Usage

TensorLite is crafted for users who need a no-frills, easy-to-set-up computational framework. It's particularly suited for educational projects, small-scale experiments, or situations where the full capabilities of TensorFlow are unnecessary. To use TensorLite, follow this basic example:

```python
from python.tensorlite import Tensor, add, multiply, matmul, Session

# Define tensors
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Set up operations
add_op = add([a, b])
mul_op = multiply([a, b])
matmul_op = matmul([a, b])

# Execute operations in a session
sess = Session()
print("Addition Result:", sess.run(add_op))
print("Multiplication Result:", sess.run(mul_op))
print("Matrix Multiplication Result:", sess.run(matmul_op))
```

## Limitations

TensorLite, while inspired by TensorFlow, does not replicate its comprehensive features and is not intended for large-scale or performance-critical applications. It serves as a tool for straightforward tensor manipulations and basic computational tasks, without the extensive functionalities and support found in more sophisticated frameworks.

## Contributions

Contributions to TensorLite are highly encouraged, as they help evolve its capabilities and utility. We welcome enhancements that maintain the framework's lightweight nature, improve usability, or extend its functionality in meaningful ways. Whether you're fixing bugs, proposing new features, or improving documentation, your input is valuable in making TensorLite more beneficial for the community.
