import torch
import struct

def shapestr(t) -> str:
    """
    Get torch tensor and return shape string of format (sizes...). E.g. (3) for length 3, (2,100)
    for 2x100 tensor.
    """
    
    return '(' + ','.join(map(str,t.shape)) + ')'

class NodeCollection:
    """Stores a collection of nodes in a Pytorch script module graph by name."""
    
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
    """Base class for converters from Pytorch script module to binary format."""
    
    def __init__(self, layer):
        
        self.source = layer
        
    def __iter__(self):
        
        return (self for _ in range(1))
        
    def __repr__(self):
        
        return f"{self.__class__.__name__}({self.source})"
        
    def convert(self, strform: str, floatform: str, intform: str) -> bytes:
        """
        Convert to bytes object. Should start with name and format string. Arguments are the string
        format to use (e.g. '64s' for 64-character string), the type of float, and the type of int.
        See https://docs.python.org/3/library/struct.html#struct-format-strings for info on how to
        specify the float and integer formats. The format string output by this function should give
        a sequence of data types and shapes. For example, f(100,100)i(1)f(20) indicates that the
        data consists of a 100x100 float array, a single integer, and then a length 20 float array.
        """
        
        raise NotImplementedError(f"'{self.__class__}' does not implement convert() method.")
        

class LinearConv(ConvBase):
    
    def __init__(self, layer):
        
        super().__init__(layer)
        
        self.weight = layer.weight
        self.bias = layer.bias
    
    def convert(self, strform, floatform, intform) -> bytes:
        
        wt, bs = self.weight, self.bias
        
        b  = struct.pack(strform, 'Linear'.encode())
        b += struct.pack(strform, f'f{shapestr(wt)}f{shapestr(bs)}'.encode())
        b += struct.pack(floatform * wt.numel(), *wt.flatten())
        b += struct.pack(floatform * bs.numel(), *bs.flatten())
        
        return b

class LeakyReluConv(ConvBase):
    
    def __init__(self, layer):
        
        super().__init__(layer)
        
        cn = layer.graph.findAllNodes('prim::Constant')
        nc = NodeCollection(cn)
        
        self.neg_slope = nc._3.f('value')
        self.inplace = bool(nc._4.i('value'))
        
    def convert(self, strform, floatform, intform) -> bytes:
        
        b  = struct.pack(strform, 'LeakyReLU'.encode())
        b += struct.pack(strform, 'f(1)'.encode())
        b += struct.pack(floatform, self.neg_slope)
        
        return b
        
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
        
        self.actual_weight = self.weight / torch.sqrt(self.running_var + self.eps)
        self.actual_bias = -self.running_mean /  torch.sqrt(self.running_var + 1e-5) * self.weight \
                + self.bias
                
    def convert(self, strform, floatform, intform) -> bytes:
        
        # Use simplified form for now
        wt = self.actual_weight
        bs = self.actual_bias
        
        b  = struct.pack(strform, 'BatchNorm1d'.encode())
        b += struct.pack(strform, f'f{shapestr(wt)}f{shapestr(bs)}'.encode())
        b += struct.pack(floatform * wt.numel(), *wt.flatten())
        b += struct.pack(floatform * bs.numel(), *bs.flatten())
        
        return b
