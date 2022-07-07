#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import struct
import layer_converters as lc

net_file_help = "PyTorch neural network file (saved using TorchScript)."
out_file_help = """Output file to write the binary network to. Default is the input filename with 
        .pnn (for Pele neural network) as the extension."""
strlen_help = "Fixed string length to use when writing the binary."
floatform_help = """Floating point format to use. Supply 'e' for half-precision, 'f' for
        single-precision, and 'd' for double-precision. Single-precision is the default."""
intform_help = """Integer format to use. Supply 'h' for 16-bit, 'i' for 32-bit, and 'l' for 64-bit.
        Unsigned integers are not allowed."""

parser = argparse.ArgumentParser()
parser.add_argument('net_file', help=net_file_help)
parser.add_argument('-o', '--out_file', default='', help=out_file_help)
parser.add_argument('-sl', '--strlen', type=int, default=64, help=strlen_help)
parser.add_argument('-ff', '--floatform', default='f', help=floatform_help)
parser.add_argument('-if', '--intform', default='i', help=intform_help)
args = parser.parse_args(sys.argv[1:])

net = torch.jit.load(args.net_file)

if not args.out_file:
    basename = os.path.basename(args.net_file)
    args.out_file = os.path.splitext(basename)[0] + '.pnn'
    
if args.floatform not in {'e', 'f', 'd'}:
    raise ValueError(f'Invalid floatform argument: {args.floatform}.')
    
if args.floatform == 'e':
    args.floatsize = 2
elif args.floatform == 'f':
    args.floatsize = 4
else:
    args.floatsize = 8
    
if args.intform.lower() == 'h':
    args.intsize = 2
elif args.intform.lower() == 'i':
    args.intsize = 4
else:
    args.intsize = 8

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

with open(args.out_file, 'wb') as f:
    
    # Write out info on string, float, and integer formats used
    f.write(struct.pack('III', args.strlen, args.floatsize, args.intsize))
    
    for c in read_net(net):
        f.write(c.convert(f'{args.strlen}s', args.floatform, args.intform))
