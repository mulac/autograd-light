import unittest
import numpy as np
from tqdm import trange
from autograd.tensor import Tensor
from autograd.other import SGD, NLLLoss, fetch_mnist, init_layer

x_train, y_train, x_test, y_test = fetch_mnist()

class Net:
    def __init__(self):
        self.l1 = Tensor(init_layer(28*28, 800))
        self.l2 = Tensor(init_layer(800, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).log_softmax()

    def parameters(self):
        return [self.l1, self.l2]


#---- INIT MODEL & HYPERPARAMS ----#
net = Net()
optim = SGD(net.parameters(), lr=1e-7)
loss_fn = NLLLoss
BS = 128

#-------------- TRAIN -------------#
for epoch in (t := trange(1000)):
    sample = np.random.randint(0, len(x_train), BS)
    X = Tensor(x_train[sample])
    Y = Tensor(y_train[sample])

    out = net.forward(X)
    loss = loss_fn(out, Y)
    loss.backward()
    optim.step()

    t.set_description('loss {}'.format(loss))

#------------ EVALULATE ------------#
out = net.forward(Tensor(x_test))
preds = out.data.argmax(axis=1)

accuracy = (preds == y_test).mean()
print("Accuracy: ", accuracy)
assert accuracy > 0.75
