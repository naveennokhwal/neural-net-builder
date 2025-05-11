import pickle
from layers import Layer
# lets define an MLP
class MLP:
    def __init__(self, nin, nout):
        self.L1 = Layer(nin,16)
        self.L2 = Layer(16,8)
        self.out = Layer(8, nout)
        self.class_to_index = None
    
    def __call__(self, x):
        x = self.L1(x)
        x = self.L2(x)
        out = self.out(x)
        return out
    
    def parameters(self):
        params = []
        params.extend(self.L1.parameters())                        
        params.extend(self.L2.parameters())                        
        params.extend(self.out.parameters())          
        return params              
    
    def save_model(model, other, path='model.pkl'):
        params = [p.data for p in model.parameters()]
        dict = {'params': params,
                'class_to_index': other}
        with open(path, 'wb') as f:
            pickle.dump(dict, f)

    def load_model(model, path='model.pkl'):
        with open(path, 'rb') as f:
            dict = pickle.load(f)

        # assign the saved data to the model's parameters
        data_values = dict['params']
        for p, val in zip(model.parameters(), data_values):
            p.data = val  

        model.class_to_index = dict['class_to_index']

