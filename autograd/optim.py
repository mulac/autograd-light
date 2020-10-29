import numpy as np
from autograd.tensor import Tensor

class SGD:
    def __init__(self, weights, lr=0.000001):
        self.weights = weights
        self.lr = lr

    def step(self):
        for w in self.weights:
            w.data += w.grad * self.lr

def NLLLoss(out, target):
    onehot = np.zeros((len(target.data), len(out.data[0])), dtype=np.float64)
    onehot[range(onehot.shape[0]), target.data] = 1
    target = Tensor(onehot)
    # -output[:, target]
    return out.mul(target).sum()
