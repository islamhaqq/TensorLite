import ctypes
import numpy as np

# Load the shared library
_lib = ctypes.CDLL('../bin/TensorLite.dll')


def vector_add(a, b):
    n = a.size  # Number of elements in the array
    out = np.empty_like(a)  # Allocate an output array
    # Define argument types and result for the C function
    _lib.vector_add.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    # Call the CUDA vector_add function
    _lib.vector_add(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n)
    )
    return out


if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    c = vector_add(a, b)
    print(c)
