import math

class Value:
    def __init__(self, data, _children=(), _op ='', label = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data = {self.data})"
    
    def __add__(a,b):
        b = b if isinstance(b, Value) else Value(b, label='const')
        out = Value(a.data + b.data, (a,b), '+')
        def _backward():
            a.grad += 1.0* out.grad
            b.grad += 1.0* out.grad

        out._backward = _backward

        return out
    
    def __sub__(a,b):
        b = b if isinstance(b, Value) else Value(b, label='const')
        out = Value(a.data - b.data, (a,b), _op = '-')
        def _backward():
            a.grad += out.grad * 1.0
            b.grad += out.grad * (-1.0)
        
        out._backward = _backward
        return out

    
    def __mul__(a,b):
        b = b if isinstance(b, Value) else Value(b, label='const')
        out = Value(a.data * b.data, (a,b), '*')
        def _backward():
            a.grad += b.data * out.grad
            b.grad += a.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(a,b):
        return a*b
    
    def __truediv__(a,b):
        # a/b = a* b**-1
        b = b if isinstance(b, Value) else Value(b, label='const')
        out = Value(a.data * b.data**-1, (a,b), _op='div')
        def _backward():
            a.grad += out.grad* b.data**-1
            b.grad += out.grad* a.data * (-1*b.data**-2)
        
        out._backward = _backward
        return out
    
    def __pow__(x, k):
        assert isinstance(k, (int, float)), "can only support int or float"
        out = Value(x.data**k, (x,), _op= f'**{k}')

        def _backward():
            x.grad += out.grad * (k* x.data**(k-1))
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1-t**2) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        x = self.data
        out = Value(max(x,0), (self,), _op = 'relu')

        def _backward():
            if x<=0:
                self.grad += out.grad * 0.0
            else:
                self.grad += out.grad* 1.0
        
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), _op= 'exp')

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward

        return out
    
    def log(self):
        x = self.data
        out = Value(math.log(x), (self,), _op= 'log')

        def _backward():
            self.grad += out.grad * (1/x)
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()