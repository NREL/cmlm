#!/usr/bin/env python3

import sys
import torch
import layer_converters as lc

net_file = sys.argv[1]
net = torch.jit.load(net_file)

def read_net(net):
    
    for m in net.children():
        f = converters.get(m.original_name.upper(), read_layer)
        yield from f(m)
        
def read_layer(l):
    
    children = list(l.children())

    for m in children:
        f = converters.get(m.original_name.upper(), read_layer)
        yield from f(m)
        
    if not children:
        yield lc.ConvBase(l)
    
converters = dict()
for attr in lc.__dict__:
    if attr[-4:] == "Conv":
        converters[attr[:-4].upper()] = getattr(lc, attr)

for c in read_net(net):
    
    print(c)
    print(c.__dict__.keys())
    print()
