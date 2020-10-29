import numpy as np


def ensure_array(arrayable):
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor():
    def __init__(self, data, children=[]):
        self.data = ensure_array(data)
        self.prev = set(children)
        self.grad = np.zeros_like(self.data, dtype=np.float)
        self.grad_fn = lambda: None

    def __repr__(self):
        return f"Tensor({self.data})"#, grad={self.grad})"

    def backward(self):
        """ Traverses the graph applying the chain rule """
        # Build the topologial graph
        graph = []
        visited = set()
        def build_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_graph(child)
                graph.append(v)
        build_graph(self)

        # Go backwards through the graph and apply the grad_fn using chain rule
        self.grad = np.ones_like(self.data, dtype=np.float)
        for v in reversed(graph):
            v.grad_fn()

    
    def add(self, other):
        ret = Tensor(self.data + other.data, [self, other])

        def _backward():
            self.grad += ret.grad
            other.grad += ret.grad
        ret.grad_fn = _backward

        return ret

    def sum(self):
        ret = Tensor(np.sum(self.data), [self])

        def _backward():
            self.grad += ret.grad
        ret.grad_fn = _backward

        return ret

    def mul(self, other):
        ret = Tensor(self.data * other.data, [self, other])

        def _backward():
            self.grad += ret.grad * other.data
            other.grad += ret.grad * self.data
        ret.grad_fn = _backward

        return ret

    def dot(self, other):
        ret = Tensor(self.data.dot(other.data), [self, other])

        def _backward():
            self.grad += ret.grad.dot(other.data.T)
            other.grad += ret.grad.T.dot(self.data).T
        ret.grad_fn = _backward

        return ret
    
    def relu(self):
        ret = Tensor(np.maximum(0, self.data), [self])

        def _backward():
            self.grad += ret.grad * (ret.data > 0)
        ret.grad_fn = _backward

        return ret

    def log_softmax(self):
        def logsumexp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
        ret = Tensor(self.data - logsumexp(self.data).reshape((-1, 1)), [self])

        def _backward():
            self.grad += ret.grad - np.exp(ret.data) * ret.grad.sum(axis=1).reshape((-1, 1))
        ret.grad_fn = _backward

        return ret

    def mean(self):
        div = Tensor(np.array([1/self.data.size], dtype=self.data.dtype))
        return self.sum().mul(div)

    # def reshape(self, *shape):
    #     ret = Tensor(self.data.reshape(shape), [self])

    #     def _backward():
    #         self.grad += ret.grad.reshape(self.data.shape)
    #     ret.grad_fn = _backward

    #     return ret
# x = np.random.randn(9).astype(np.float32)
# W = np.random.randn(9, 9).astype(np.float32)


# tx = Tensor(x)
# tW = Tensor(W)

# t0 = tx.add(tx)
# t1 = t0.dot(tW)
# t2 = t1.relu()
# t3 = t2.sum()
# t3.backward()
# print(t3)
# print(t2)
# print(t1)
# print(t0)
# print(tx)
"""

def register(name, function):
    setattr(Tensor, name, function)


def add(self, other):
    assert isinstance(other, Tensor)
    ret = Tensor(self.data + other.data, [self, other])

    def _backward():
        self.grad += ret.grad
        other.grad += ret.grad
    ret.grad_fn = _backward

    return ret
register('add', add)

def sum(self):
    ret = Tensor(np.sum(self.data), [self])

    def _backward():
        self.grad += ret.grad
    ret.grad_fn = _backward

    return ret
register('sum', sum)

def dot(self, other):
    ret = Tensor(self.data.dot(other.data), [self, other])

    def _backward():
        self.grad += ret.grad * other.data *2
        other.grad += ret.grad * self.data *2
    ret.grad_fn = _backward

    return ret
register('dot', dot)


class Function():
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return Tensor(x.data.sum())
    
    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors
        return x.data*grad
register('sum', Sum)

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return Tensor(x.data * other.data)
    
    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        return y*grad, x*grad
register('mul', Mul)

"""
