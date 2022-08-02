import torch
from torch import nn
from collections import OrderedDict

def repeat_value(val, max_times=float("inf")):
    
    count = 0
    while count < max_times:
        yield val
        count += 1
        
class PredictionSubnet(torch.nn.Module):
    
    def __init__(self, nin, hsizes, nout=1):
        
        super().__init__()
        
        self.nin = nin
        self.nh = len(hsizes)-1
        self.nout = nout
        self.hsizes = hsizes
        
        self.inp = nn.Linear(nin, hsizes[0])
        
        layers = OrderedDict()
        for i in range(self.nh):
            layers["relu" + str(i)] = nn.LeakyReLU()
            layers["bn" + str(i)] = nn.BatchNorm1d(hsizes[i])
            layers["linear" + str(i)] = nn.Linear(hsizes[i], hsizes[i+1])
        self.hidden = nn.Sequential(layers)

        self.output = nn.Sequential(nn.LeakyReLU(), nn.BatchNorm1d(hsizes[-1]),
                nn.Linear(hsizes[-1], nout))
        
    def forward(self, xin):
        
        out = self.inp(xin)
        out = self.hidden(out)
        out = self.output(out)
        return out

class NetArray(torch.nn.Module):
    """
    Represent a large neural net with an array of disconnected subnets.
    """

    def __init__(self, nin, nets=(), nouts=(), freeze_structure=False):
        
        super().__init__()

        assert nin > 0
        self._nin = nin
        self._nout = 0
        self._nets = []
        self._nouts = []
        self._evaluable = False
        
        nouts = tuple(nouts)
        if len(nouts) == 0:
            nout_gen = repeat_value(1)
        elif len(nouts) == 1:
            nout_gen = repeat_value(nouts[0])
        else:
            nout_gen = iter(nouts)
            
        for net, nout in zip(nets, nout_gen):
            self.add(net, nout=nout)
            
        if freeze_structure:
            self.freeze_structure()

    def __len__(self):

        return len(self._nets)
        
    def __iter__(self):
        
        return iter(self._nets)

    def __getitem__(self, idx):

        return self._nets[idx]
        
    def __setitem__(self, idx, value):
        
        self._assert_unfrozen()
        
        try:
            net, nout = value
        except TypeError:
            net = value
            nout = 1
        self._nets[idx] = net
        self._nout += nout - self._nouts[idx]
        self._nouts[idx] = nout
        
        setattr(self, self._subnet_name(idx), net)

    def __call__(self, xin):

        return self.forward(xin)
        
    def _assert_unfrozen(self):
        
        assert (not self._evaluable), "Cannot modify subnet list for" + \
                f" {self.__class__.__name__} that has been frozen."
        
    def _subnet_name(self, idx):
        
        return f"net{idx}"
        
    def add(self, net, idx=None, nout=1):
        
        self._assert_unfrozen()
        
        if idx is None:
            idx = len(self)
        self._nets.insert(idx, net)
        self._nouts.insert(idx, nout)
        self._nout += nout
        
        self.add_module(self._subnet_name(idx), net)
        
    def remove(self, idx=None, nout=1):
        
        self._assert_unfrozen()
        
        if idx is None:
            idx = len(self)
        self._nets.pop(idx)
        self._nouts.pop(idx)
        self._nout -= nout
        
        delattr(self, self._subnet_name(idx))
        
    def freeze_structure(self):
        
        self.register_buffer('output', torch.zeros(self.nout), False)
        self._evaluable = True
        
    def unfreeze_structure(self):
        
        del self.output
        self._evaluable = False
        
    @property
    def nin(self):
        
        return self._nin
        
    @property
    def nout(self):
        
        return self._nout

    def forward(self, xin):
        
        assert self._evaluable, "Must call freeze_structure() before evaluating" + \
                f" {self.__class__.__name__}!"
        
        iout = 0

        for i in range(len(self)):
            
            net = self._nets[i]
            nout = self._nouts[i]
            self.output[iout:iout+nout] = net(xin)
            iout += nout
            
        return self.output
        
if __name__ == "__main__":
    
    netarr = NetArray(2, nets=[PredictionSubnet(2, [5, 5])])
    netarr.add(PredictionSubnet(2, [5, 5]))
    netarr.freeze_structure()
    netarr.eval()
    print(netarr(torch.tensor([[1., 2.]])))
