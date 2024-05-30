# Implementation of https://github.com/karpathy/micrograd.git for Tensors

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = []
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
    
    
    def __mul_scalar(self, scalar: float):
        ...
        
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        