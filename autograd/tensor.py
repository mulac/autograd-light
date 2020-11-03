# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
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
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self.grad_fn = lambda: None

    def __repr__(self):
        return f"Tensor({self.data}), \
                grad={self.grad}, \
                grad_fn={self.grad_fn.__name__})"

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
        self.grad = np.ones_like(self.data, dtype=np.float32)
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
