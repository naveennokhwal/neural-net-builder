import random
from value import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1), label='W') for _ in range(nin)]
        self.b = Value(random.uniform(-1,1), label='b')
    
    def __call__(self,x):
        act =  sum((wi*xi for wi,xi in zip(self.w, x)), self.b) 
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()] 
    