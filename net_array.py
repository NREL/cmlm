import torch
import copy
import os
from torch import nn
from collections import OrderedDict

def repeat_value(val, max_times=float("inf")):
    
    count = 0
    while count < max_times:
        yield val
        count += 1

class NetArray(torch.nn.Module):
    """
    Represent a large neural net with an array of disconnected subnets.
    """

    def __init__(self, nets=(), nouts=(), **kwargs):
        
        super().__init__()
        
        self.net_type = "NetArray"

        self._nout = 0
        self._nets = []
        self._nouts = []
        
        nouts = tuple(nouts)
        if len(nouts) == 0:
            nout_gen = repeat_value(1)
        elif len(nouts) == 1:
            nout_gen = repeat_value(nouts[0])
        else:
            nout_gen = iter(nouts)
            
        for net, nout in zip(nets, nout_gen):
            self.add(net, nout=nout)
            
        self.inputs = dict()
        self.inputs['nmv'] = kwargs.pop('nmv', None)
        self.inputs['npass'] = kwargs.pop('npass', None)
        self.inputs['manidef'] = kwargs.pop('manidef', None)
        self.inputs['manibiases'] = kwargs.pop('manibiases', None)
        self.inputs['manibiases_supplied'] = kwargs.pop('manibiases_supplied', True) and \
                self.inputs['manibiases'] is not None
        self.inputs['nvar'] = kwargs.pop('nvar', None)
        if self.inputs['manidef'] is not None:
            self.inputs['D_in'] = self.inputs['manidef'].shape[0]
            if 'D_in' in kwargs:
                assert kwargs['D_in'] == self.inputs['D_in']
                del kwargs['D_in']
        else:
            self.inputs['D_in'] = kwargs.pop('D_in', None)
        
        if self.inputs['manidef'] is not None:
            self.inputs['manidef'] = torch.as_tensor(self.inputs['manidef'], dtype=torch.float)
            if self.inputs['manibiases'] is None:
                self.inputs['manibiases'] = torch.zeros(
                        self.inputs['manidef'].shape[1], dtype=torch.float)
                        
        if self.inputs['nvar'] is not None:
            self.calc_manifold = self.calc_manifold_wvar
        else:
            self.calc_manifold = self.calc_manifold_novar
            
        if kwargs.pop('check_kwargs', True):
            assert (not kwargs), f"Unrecognized keyword arguments: {kwargs}"

    def __len__(self):

        return len(self._nets)
        
    def __iter__(self):
        
        return iter(self._nets)

    def __getitem__(self, idx):

        return self._nets[idx]
        
    def __setitem__(self, idx, value):
        
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
        
    def _subnet_name(self, idx):
        
        return f"net{idx}"
        
    def nout_iter(self):
        
        return iter(self._nouts)
        
    def add(self, net, idx=None, nout=1):
        
        if idx is None:
            idx = len(self)
        self._nets.insert(idx, net)
        if hasattr(net, 'nout'):
            nout = net.nout
        self._nouts.insert(idx, nout)
        self._nout += nout
        
        self.add_module(self._subnet_name(idx), net)
        
    def remove(self, idx=None, nout=1):
        
        if idx is None:
            idx = len(self)-1
        del self._nets[idx]
        del self._nouts[idx]
        self._nout -= nout
        
        for i in range(idx, len(self)+1):
            delattr(self, self._subnet_name(i))
        for i in range(idx, len(self)):
            setattr(self, self._subnet_name(i), self._nets[i])
        
    def get_manvar(self, x):
        
        return torch.matmul(x, self.inputs['manidef']) + self.inputs['manibiases']
        
    def calc_manifold_novar(self, x):
        
        D_in = self.inputs['D_in']
        out1 = self.get_manvar(x[:,:D_in])
        out3 = x[:,D_in:]
        return torch.cat((out1, out3), 1)

    def calc_manifold_wvar(self, x):
        
        D_in = self.inputs['D_in']
        out1 = self.get_manvar(x[:,:D_in])

        # Manifold variable variances (xi_var)
        out2 = x[:,D_in:D_in + D_in**2].reshape(-1,D_in,D_in)
        out2 = self.get_manvar(out2).transpose(1,2)
        out2 = self.get_manvar(out2)
        out2 = torch.diagonal(out2,dim1=1,dim2=2) - out1**2
        out2 = out2[:,:self.inputs['nvar']]

        # Passed through variables (passvars)
        out3 = x[:,D_in+D_in**2:]
        return torch.cat((out1, out2, out3),1)
        
    def _unscaled_manidef(self, scalers):
        
        if 'manidef' not in self.inputs:
            return None, None
        
        manidef = (self.inputs['manidef'].T / torch.Tensor(scalers['inp'].scale_)).T
        if not (scalers['inp'].with_mean):
            manibiases = torch.zeros_like(self.inputs['manibiases'])
        else:
            manibiases = -torch.matmul(self.inputs['manidef'].T, torch.Tensor(scalers['inp'].mean_))
        
        return manidef, manibiases
    
    def unscaled(self, scalers, compute_mani_source=False, src_term_map=None, src_net_idx=None):
        
        kwargs = dict(self.inputs)
        kwargs['manidef'], kwargs['manibiases'] = self._unscaled_manidef(scalers)
        
        if compute_mani_source:
            assert src_term_map is not None
            assert src_net_idx is not None
        else:
            src_net_idx = len(self) + 1
        
        unsc_arr = self.__class__(**kwargs)
        
        outpos = 0
        for i in range(len(self)):
            outslice = slice(outpos, outpos+self._nouts[i])
            outpos += self._nouts[i]
            if i == src_net_idx:
                unsc_net = self[i].unscaled(scalers, compute_mani_source, src_term_map, outslice)
                unsc_arr.add(unsc_net, nout=self._nouts[i]+self.inputs['nmv'])
            else:
                unsc_arr.add(self[i].unscaled(scalers, outslice=outslice), nout=self._nouts[i])
                
        return unsc_arr
            
    @property
    def nin(self):
        
        if self.inputs['nmv'] is None or self.inputs['npass'] is None:
            return None
        return self.inputs['nmv'] + self.inputs['npass']
        
    @property
    def nout(self):
        
        return self._nout

    def forward(self, xin):
        
        iout = 0        
        dev = next(self.parameters()).device
        output = torch.zeros((xin.shape[0], self.nout), dtype=torch.float, device=dev)

        for i in range(len(self)):
            
            net = self._nets[i]
            nout = self._nouts[i]
            output[:, iout:iout+nout] = net(xin)
            iout += nout
            
        return output
        
    def save_torchscript(self, dirname):
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for i in range(len(self)):
            torch.jit.script(self[i]).save(os.path.join(dirname, self._subnet_name(i) + ".pt"))
        
class MRNetArray(NetArray):
    """Container for NetArray that allows for learning of manifold reduction matrix."""
    
    def __init__(self, mrmodule=None, nets=(), nouts=(), **kwargs):
        
        super().__init__(nets, nouts, check_kwargs=True, **kwargs)
        
        if mrmodule is not None:
            self.manifold = mrmodule
        elif self.inputs['manidef'] is not None:
            if self.inputs['manibiases_supplied']:
                self.manifold = nn.Linear(self.inputs['D_in'], self.inputs['nmv'], bias=True)
                self.manifold.weight = nn.Parameter(torch.as_tensor(self.inputs['manidef'].T,
                        dtype=torch.float))
                self.manifold.bias = nn.Parameter(torch.as_tensor(self.inputs['manibiases'],
                        dtype=torch.float))
            else:
                self.manifold = nn.Linear(self.inputs['D_in'], self.inputs['nmv'], bias=False)
                self.manifold.weight = nn.Parameter(torch.as_tensor(self.inputs['manidef'].T,
                        dtype=torch.float))
        else:
            self.manifold = nn.Linear(self.inputs['D_in'], self.inputs['nmv'], bias=False)
            
        self.inputs.pop('manidef', None)
        self.inputs.pop('manibiases', None)
        
        self.net_type = "MRNetArray"
        
    def get_manvar(self, x):
        
        return self.manifold(x)
        
    def _unscaled_manidef(self, scalers):
        
        sdict = self.state_dict()
        
        manidef = (sdict['manifold.weight'] / torch.Tensor(scalers['inp'].scale_)).T
        if not (scalers['inp'].with_mean):
            manibiases = torch.zeros_like(sdict['manifold.bias'])
        else:
            manibiases = -torch.matmul(sdict['manifold.weight'], torch.Tensor(scalers['inp'].mean_))
        
        return manidef, manibiases
        
    def forward(self, xin):
        
        xin = self.calc_manifold(xin)
        return super().forward(xin)
        
    @classmethod
    def from_netarray(cls, netarray):
        
        new_subnets = [net.copy_state() for net in netarray]
        return cls(nets=new_subnets, nouts=list(netarray.nout_iter()), **netarray.inputs)
        
    def to_netarray(self):
        
        kwargs = dict(self.inputs)
        sdict = self.state_dict()
        
        kwargs['manidef'] = copy.deepcopy(sdict['manifold.weight']).T
        if 'manifold.bias' in sdict:
            kwargs['manibiases'] = copy.deepcopy(self.state_dict()['manifold.bias'])
        new_subnets = [net.copy_state() for net in self]
        return NetArray(new_subnets, list(self.nout_iter()), **kwargs)
        
if __name__ == "__main__":
    
    from manifold_reduction_nets import PredictionNet
    
    netarr = NetArray(nets=[PredictionNet(2, [5, 5], 1)], nmv=2)
    netarr.add(PredictionNet(2, [5, 5], 1))
    netarr.add(PredictionNet(2, [5, 5], 1))
    netarr.add(PredictionNet(2, [5, 5], 1))
    netarr.eval()
    print(netarr(torch.tensor([[1., 2.], [3., 4.]])))
