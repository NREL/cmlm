import pydoc
import sys
import torch

def output_help_to_file(filepath, request):
    
    f = open(filepath, 'w')
    sys.stdout = f
    pydoc.help(request)
    f.close()
    sys.stdout = sys.__stdout__
    
class NodeCollection:
    
    def __init__(self, nodes):
        
        self._nodes_ = nodes
        self._keys_ = []
        
        for node in nodes:
            
            node_name = str(node).split(':')[0].strip()
            node_name = node_name.lstrip('%')
            if node_name.isnumeric():
                node_name = '_' + node_name
            setattr(self, node_name, node)
            
            self._keys_.append(node_name)
            
    def __iter__(self):
        
        return iter(self._nodes_)
        
    def __len__(self):
        
        return len(self._nodes_)
        
    def __getitem__(self, key):
        
        return getattr(self, key)
        
    def __repr__(self):
        
        keystr = ', '.join(self.key_list_[:3])
        if len(self) > 3:
            keystr += ", ..."
        return f"NodeCollection(len={len(self)}, keys=[{keystr}])"
        
    def print_nodes_(self):
        
        print("NodeCollection:\n")
        for node in self:
            print('', node)
        
    @property
    def keys_(self):
        
        return iter(self._keys_)
        
    @property
    def values_(self):
        
        return iter(self._nodes_)
        
    @property
    def items_(self):
        
        return zip(self._keys_, self._nodes_)
        
    @property
    def key_list_(self):
        
        return self._keys_
    
    @property    
    def value_list_(self):
        
        return self._nodes_
    
    @property    
    def item_list_(self):
        
        return list(self.items_)

class ConvBase:
    
    def __init__(self, layer):
        
        self.source = layer
        
    def __iter__(self):
        
        return (self for _ in range(1))
        
    def __repr__(self):
        
        return f"{self.__class__.__name__}({self.source})"
        
    def convert(self):
        
        raise NotImplementedError(f"'{self.__class__}' does not implement convert() method.")
        

class LinearConv(ConvBase):
    
    def __init__(self, layer):
        
        super().__init__(layer)
        
        self.weight = layer.weight
        self.bias = layer.bias
        

class LeakyReluConv(ConvBase):
    
    def __init__(self, layer):
        
        super().__init__(layer)
        
        cn = layer.graph.findAllNodes('prim::Constant')
        nc = NodeCollection(cn)
        
        self.neg_slope = nc._3.f('value')
        self.inplace = bool(nc._4.i('value'))
        
class BatchNorm1dConv(ConvBase):
    
    def __init__(self, layer):
        
        super().__init__(layer)
        
        self.training = layer.training
        self.running_mean = layer.running_mean
        self.running_var = layer.running_var
        self.weight = layer.weight
        self.bias = layer.bias
        
        nc = NodeCollection(layer.graph.findAllNodes('prim::Constant'))
        self.momentum = nc._39.f('value')
        self.eps = nc._40.f('value')
        
        self.lin_weight = self.weight / torch.sqrt(self.running_var + self.eps)
        self.lin_bias = - self.running_mean /  torch.sqrt(self.running_var + 1e-5) * self.weight \
                + self.bias
