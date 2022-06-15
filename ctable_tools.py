import struct
import sys
import numpy as np
import pandas as pd

def convert_chemtable_units(ctable,conversion='mks2cgs'):
    # MKS to CGS conversion factors
    conversions = {'RHO':1.0e-3, #  kg m-3 -> g cm-3
                   'DIFF':10.0, # (rhoD) kg s-1 m-1 -> g s-1 cm-1 
                   'VISC':10.0, # (dynamic) kg s-1 m-1 -> g s-1 cm-1
                   'SRC_':1.0e-3, # source terms kg m-3 s-1 -> g cm-3 s-1
                   'T':1.0, # K -> K
                   'X':1.0e2, # m -> cm
                   'VEL':1.0e2, } # m s-1 -> cm s-1
    
    if conversion not in ['mks2cgs','cgs2mks','custom']:
        raise RuntimeError("convert_chemtable_units: can only convert mks2cgs or cgs2mks")
    
    if conversion == 'cgs2mks':
        for var in conversions.keys():
            conversions[var] = 1/conversions[var]

    for var in ctable.columns:
        varmod = var if not var.startswith('SRC_') else 'SRC_'
        if varmod in conversions.keys():
            ctable[var] *= conversions[varmod]

        elif not var.startswith('Y-'):
            # Warn if not a mass fraction and no conversion is found
            print('WARNING: no conversion for tabulated variable {}'.format(var))
            
    return 0


def print_chemtable(ctable):
    print()
    print("--- TABULATED FUNCTION ---")
    print()
    print("Dimensions ({}):".format(len(ctable.index.names)))
    for ii, dim in enumerate(ctable.index.names):
        print('    Dim: {:<2d} Name: {:<10s} Length: {}'.format(ii, dim, len(ctable.index.levels[ii])))
        print('         Values:')
        print(' '+' '.join([('         ' if ii%4==0 else '') +
                            "{:16.8e}".format(val)
                            + ('\n' if ii%4 ==3 else '')
                            for ii,val in enumerate(ctable.index.levels[ii])]))
    print()
    print("Variables ({}):".format(len(ctable.columns)))
    for ii,var in enumerate(ctable.columns):
        print('    Var: {:<2d} Name: {:<10s} Min: {:16.8e} Max: {:16.8e}'.format(ii, var, np.min(ctable[var]), np.max(ctable[var])))
    print()
    
def read_chemtable_binary(filename, tformat='Pele', Ndim=None, verbose=0):
    # check inputs
    if tformat not in ["Pele", "NGA"]:
        raise RuntimeError("Invalid table format")
    if tformat == "NGA":
        if Ndim is None:
            raise RuntimeError("Must specify number of dimensions for NGA format")

    with open(filename, 'rb') as fi:
        if tformat == "Pele":
            Ndim = struct.unpack('i', fi.read(4))[0]
            dim_names =[struct.unpack('64s', fi.read(64))[0].decode().strip() for idim in range(Ndim)]
        else:
            dim_names =['dim'+str(idim) for idim in range(Ndim)]
        
        dimLengths = struct.unpack(str(Ndim)+'i', fi.read(Ndim*4))
        Nvar = struct.unpack('i', fi.read(4))[0]
        grids = [struct.unpack(str(dimLengths[idim])+'d', fi.read(dimLengths[idim]*8)) for idim in range(Ndim)]
        model_name = struct.unpack('64s', fi.read(64))[0].decode().strip()
        var_names =[str(struct.unpack('64s', fi.read(64))[0].decode().strip()) for ivar in range(Nvar)]
        Ndata_var = np.product(list(dimLengths))
        Ndata = Nvar * Ndata_var
        data = np.array(struct.unpack(str(Ndata)+'d', fi.read(Ndata*8))).reshape(Ndata_var,Nvar,order='F')

    ctable_index = pd.MultiIndex.from_product(reversed(grids),names=reversed(dim_names))
    ctable = pd.DataFrame(data,index=ctable_index,columns=var_names)
    
    if verbose > 0:
        print_chemtable(ctable)
        
    if verbose> 1:
        print(ctable)        
    
    return ctable, model_name

def write_chemtable_binary(filename, ctable, tablename, tformat='Pele'):
    
    with open(filename, 'wb') as fi:

        Ndim = len(ctable.index.names)
        if tformat == "Pele":
            # Number of Dimensions
            fi.write(struct.pack('i',Ndim))
            # Dimension Names
            fi.write(struct.pack(str(Ndim*64)+'s',
                                 ''.join(['{:<64s}'.format(name)
                                          for name in reversed(ctable.index.names)]).encode()))

        # Dimension Lengths
        fi.write(struct.pack(str(Ndim)+'i', *[len(level) for level in reversed(ctable.index.levels)]))

        # Number of Variables
        fi.write(struct.pack('i', len(ctable.columns)))

        # Grids
        for ii in reversed(range(Ndim)):
            fi.write(np.array(ctable.index.levels[ii]).tobytes())

        # Model Name
        fi.write(struct.pack('64s',tablename.encode()))
        
        # Variable Names
        fi.write(struct.pack(str(len(ctable.columns)*64)+'s',
                             ''.join(['{:<64s}'.format(name)
                                      for name in ctable.columns]).encode()))

        # Data
        for col in ctable.columns:
            fi.write(ctable[col].to_numpy().tobytes())

    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Useful tools for interacting with chemtable files')
    parser.add_argument('inputfile',help='Table file to read', type=str)
    parser.add_argument('-f','--format', dest="tformat", choices=['NGA','Pele'], default='Pele',
                        help='Table format, NGA or Pele')
    parser.add_argument('-d','--dimensions',type=int,
                        help='Number of dimensions in table') 
    parser.add_argument('-p','--print', dest="print_table", action='store_true',
                        help='Flag to print extrma of table file')
    parser.add_argument('-o','--outputfile', type=str,
                        help='Output file to write, if needed')
    parser.add_argument('-c','--convert',action='store_true',
                        help='Convert format from input format to other format')
    args = parser.parse_args()

    ctable, tname = read_chemtable_binary(args.inputfile, args.tformat, args.dimensions)

    if args.print_table:
        print_chemtable(ctable)
        print(ctable)

    if args.convert:
        if args.outputfile == None: raise RuntimeError("Output file must be specified for table format conversion")
        write_chemtable_binary(args.outputfile, ctable, tname, "Pele" if args.tformat == "NGA" else "NGA")
        
    
