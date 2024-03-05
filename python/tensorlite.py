import ctypes
import numpy as np

# Load the shared library
_lib = ctypes.CDLL('../bin/TensorLite.dll')


class Tensor:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)


def add(inputs):
    a, b = inputs
    result = np.empty_like(a.data)
    _lib.vector_add(result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    a.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    b.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int(a.data.size))
    return Tensor(result)


def multiply(inputs):
    a, b = inputs
    result = np.empty_like(a.data)
    _lib.vector_multiply(result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         a.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         b.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         ctypes.c_int(a.data.size))
    return Tensor(result)


def matmul(inputs):
    a, b = inputs
    assert a.data.shape == b.data.shape, "Matrices must have the same dimensions"
    n = a.data.shape[0]  # Assuming a is a square matrix
    result = np.empty((n, n), dtype=np.float32)  # Allocate result matrix

    _lib.matrixMul(a.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   b.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   ctypes.c_int(n))

    return Tensor(result)


class Session:
    def run(self, operation):
        return operation.data
