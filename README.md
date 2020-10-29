# autograd_mini
A lightweight implementation of pytorch's autograd.  For education purposes.  Inspired by Karpathy and Hotz.

It includes enough functions to create and train a fully connected neural network, from nothing but numpy.  It uses the principle of automatic differentiation which underlies all widely used deep learning frameworks today. 

Builds upon [karpathy/micrograd](https://github.com/karpathy/micrograd). View [geohot/tinygrad](https://github.com/geohot/tinygrad) to see how you would expand on this further and then of course [pytorch](https://github.com/pytorch/pytorch).

### Installation
  ```pip install autograd```
  
### Example 
```python
from autograd.tensor import Tensor
from autograd.other import SGD, NLLLoss, fetch_mnist, init_layer

class Net:
    def __init__(self):
        self.l1 = Tensor(init_layer(28*28, 800))
        self.l2 = Tensor(init_layer(800, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).log_softmax()

    def parameters(self):
        return [self.l1, self.l2]
        
net = Net()
optim = SGD(net.parameters(), lr=1e-7)
loss_fn = NLLLoss
BS = 128
```
