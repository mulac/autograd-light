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

def init_layer(*shape):
    return (np.random.uniform(-1., 1., size=shape) / np.sqrt(np.prod(shape))).astype(np.float32)

def fetch(url):
    import requests, os, hashlib, gzip, numpy
    path = os.path.join('/tmp', hashlib.md5(url.encode()).hexdigest())
    
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = f.read()
    else:
        with open(path, 'wb') as f:
            data = requests.get(url).content
            f.write(data)

    return numpy.frombuffer(gzip.decompress(data), dtype=numpy.uint8)

def fetch_mnist():
    x_train = fetch('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')[0x10:].reshape((-1, 28*28)).astype(np.float32)
    y_train = fetch('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')[8:]
    x_test = fetch('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')[0x10:].reshape((-1, 28*28)).astype(np.float32)
    y_test = fetch('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')[8:]

    return x_train, y_train, x_test, y_test
