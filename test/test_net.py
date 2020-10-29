import torch
import unittest
import numpy as np
from tqdm import trange
from autograd.tensor import Tensor
from autograd.optim import SGD, NLLLoss


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

x_train = fetch('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')[0x10:].reshape((-1, 28*28)).astype(np.float32)
y_train = fetch('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')[8:]
x_test = fetch('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')[0x10:].reshape((-1, 28*28)).astype(np.float32)
y_test = fetch('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')[8:]


class Net:
    def __init__(self):
        self.l1 = Tensor((np.random.uniform(-1., 1., size=(28*28, 800))/np.sqrt(28*28*800)).astype(np.float32))
        self.l2 = Tensor((np.random.uniform(-1., 1., size=(800,10))/np.sqrt(800*10)).astype(np.float32))

    def forward(self, x):
        out = x.dot(self.l1).relu().dot(self.l2).log_softmax()
        return out

    def parameters(self):
        return [self.l1, self.l2]


class TestNet(unittest.TestCase):
    net = Net()
    optim = SGD(net.parameters(), lr=1e-7)
    loss_fn = NLLLoss
    BS = 128

    # TRAIN
    for epoch in (t := trange(1000)):
        sample = np.random.randint(0, len(x_train), BS)
        X = Tensor(x_train[sample])
        Y = Tensor(y_train[sample])

        out = net.forward(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optim.step()

        t.set_description('loss {}'.format(loss))

    # EVALULATE
    out = net.forward(Tensor(x_test))
    preds = out.data.argmax(axis=1)
    
    accuracy = (preds == y_test).mean()
    print(accuracy)
    assert accuracy > 0.75
