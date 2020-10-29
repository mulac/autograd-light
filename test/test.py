import unittest
import numpy as np
import torch
import autograd.tensor as tensor


x_init = np.random.randn(1,3).astype(np.float32) /10000
W_init = np.random.randn(3,3).astype(np.float32) /10000
m_init = np.random.randn(1,3).astype(np.float32) /10000

class TestTensor(unittest.TestCase):

    def test_operations(self):
        def test_minitorch():
            x = tensor.Tensor(x_init)
            W = tensor.Tensor(W_init)
            m = tensor.Tensor(m_init)

            out = x.dot(W).relu()
            out = out.log_softmax()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.data, x.grad, W.grad

        def test_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            m = torch.tensor(m_init)

            out = x.matmul(W).relu()
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, W.grad

        for expected, actual in zip(test_pytorch(), test_minitorch()):
            np.testing.assert_allclose(actual, expected, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
