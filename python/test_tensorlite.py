import unittest
import numpy as np

from tensorlite import Tensor, add, multiply, matmul, Session


class MyTestCase(unittest.TestCase):
    def test_add(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        expected_addition = [5, 7, 9]

        add_op = add([a, b])
        sess = Session()
        addition_result = sess.run(add_op)

        self.assertTrue(np.allclose(addition_result, expected_addition),
                        f"Addition Result incorrect: got {addition_result}, expected {expected_addition}")

    def test_multiply(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        expected_multiplication = [4, 10, 18]

        mul_op = multiply([a, b])
        sess = Session()
        multiplication_result = sess.run(mul_op)

        self.assertTrue(np.allclose(multiplication_result, expected_multiplication),
                        f"Multiplication Result incorrect: got {multiplication_result}, expected {expected_multiplication}")

    def test_matmul(self):
        # Identity matrices for simplicity
        a = Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        b = Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        expected_matrix_multiplication = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Result of identity matrix multiplied by itself

        matmul_op = matmul([a, b])
        sess = Session()
        matrix_multiplication_result = sess.run(matmul_op)

        self.assertTrue(np.allclose(matrix_multiplication_result, expected_matrix_multiplication),
                        f"Matrix Multiplication Result incorrect: got {matrix_multiplication_result}, expected {expected_matrix_multiplication}")


if __name__ == '__main__':
    unittest.main()
