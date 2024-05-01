import struct
import sys
import os
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
    
    if conversion not in ['mks2cgs','cgs2mks']:
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
    
def read_chemtable_binary(filename, tformat='Pele', Ndim=None, Dimnames=None, verbose=0):
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
            if Dimnames is None:
                dim_names =['dim'+str(idim) for idim in range(Ndim)]
            else:
                if len(Dimnames) != Ndim: raise RuntimeError("Wrong number of dim names given")
                dim_names = Dimnames
        
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
            assert ctable.dtypes[col] == np.float64, "Chemtable data type must be float64"
            fi.write(ctable[col].to_numpy().tobytes())

def slice_table(ctable, slice_vars=None, slice_vals=None, slice_pairs=None):
    # create a slice of a chemtable
    # must specify lists of slice_vars and slice_vals or slice_pairs of format ["var0:val0", "var1:val1", "var2:val2"]
    # slices in dimensions specified by slice_vars at locations specified by slice_vals
    if slice_pairs is not None:
        assert(slice_vars is None and slice_vals is None)
        slice_vars = [pair.split(':')[0] for pair in slice_pairs]
        slice_vals = [float(pair.split(':')[1]) for pair in slice_pairs]
    else:
        assert(slice_vars is not None and slice_vals is not None)
    assert(len(slice_vars) == len(slice_vals))
    slice_val_dict = dict(zip(slice_vars,slice_vals)) 
    found_vars = []
    found_slice_vals = []
    for var in slice_vars:
        assert(var in ctable.index.names)
        assert(var not in found_vars) # no repeats allowed
        found_vars.append(var)
        vals = ctable.index.get_level_values(var)
        closest_val = vals[np.argmin(np.abs(vals - slice_val_dict[var]))]
        print('Slice variable ' + var + ' at ' + str(closest_val) + ' (requested ' + str(slice_val_dict[var]) + ')')
        found_slice_vals.append(closest_val)
    sliced_table = ctable[ctable.index.get_loc_level(found_slice_vals,slice_vars)[0]].reset_index(level=slice_vars, drop=True)
    try:
        fake = (sliced_table.index.nlevels > 1)
    except AttributeError:
        sliced_table.index = pd.MultiIndex.from_arrays([sliced_table.index])
    return sliced_table
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Useful tools for interacting with chemtable files')
    parser.add_argument('inputfile',help='Table file to read', type=str)
    parser.add_argument('-f','--format', dest="tformat", choices=['NGA','Pele'], default='Pele',
                        help='Table format, NGA or Pele')
    parser.add_argument('-d','--dimensions',type=int,
                        help='Number of dimensions in NGA table')
    parser.add_argument('-dl','--dimension_labels', type=str, nargs='+',
                        help='List of dimension names in NGA table') 
    parser.add_argument('-p','--print', dest="print_table", action='store_true',
                        help='Flag to print extrema of table file')
    parser.add_argument('-o','--outputfile', type=str,
                        help='Output file to write, if needed')
    parser.add_argument('-cf','--convert_format',action='store_true',
                        help='Convert format from input format to other format')
    parser.add_argument('-cu','--convert_units',type=str, choices=['none','cgs2mks','mks2cgs'], default='none',
                        help='Convert units and save new table')
    parser.add_argument('-sl','--slice',type=str, nargs='+',
                        help='''Create a table by slicing. Specify dimensions to slice and values as a list of form
                        dim_name1:value dim_name2:value''')
    parser.add_argument('-sp','--slice_plot', type=str, nargs='+',
                        help='''Plot a slice of the table. Specify dimensions to slice and values as a list of form
                        dim_name1:value dim_name2:value, must specify enough slice dims such that there are exactly
                        one or two slice dimensions''')
    parser.add_argument('-v','--variables', type=str, nargs='+',
                        help='variables to be plotted')
    args = parser.parse_args()

    ctable, tname = read_chemtable_binary(args.inputfile, args.tformat, args.dimensions, args.dimension_labels)

    if args.print_table:
        print_chemtable(ctable)
        print(ctable)

    convert = (args.convert_format
               or (args.convert_units != 'none')
               or args.slice is not None)
    if convert:
        if args.outputfile == None:
            raise RuntimeError("Output file must be specified for table format/units/slicing conversion")
        
        if args.slice is not None:
            out_table = slice_table(ctable, slice_pairs=args.slice)
        else:
            out_table = ctable
            
        if args.convert_units != 'none':
            convert_chemtable_units(out_table, conversion=args.convert_units)

        if args.convert_format:
            output_format = "Pele" if args.tformat == "NGA" else "NGA"
        else :
            output_format = args.tformat

        write_chemtable_binary(args.outputfile, out_table, tname, output_format)

    if args.slice_plot is not None:
        plot_dims = ctable.index.nlevels - len(args.slice_plot)
        assert(plot_dims > 0 and plot_dims <= 2)
        assert(args.outputfile is not None)
        if not (os.path.exists(args.outputfile)): os.makedirs(args.outputfile)
        plt_vars = args.variables if args.variables is not None else ctable.columns
        sliced = slice_table(ctable, slice_pairs=args.slice_plot)
        
        import matplotlib.pyplot as plt
        if plot_dims == 1:
            for var in plt_vars:
                plt.figure()
                plt.plot(sliced.index.values, sliced[var],'r-')
                plt.xlabel(sliced.index.names[0])
                plt.ylabel(var)
                plt.savefig(os.path.join(args.outputfile,
                                         '1Dslice_' + "_".join([pair.replace(":","") for pair in args.slice_plot])
                                         + "_" + var + "_vs_" + sliced.index.names[0] + ".png"))
                plt.clf()
                plt.close()
        elif plot_dims == 2:
            for var in plt_vars:
                plt.figure()
                plt.contourf(sliced.index.levels[0],
                             sliced.index.levels[1],
                             sliced[var].to_numpy().reshape(sliced.index.levshape),
                             levels=100)
                plt.xlabel(sliced.index.names[0])
                plt.ylabel(sliced.index.names[1])
                plt.colorbar(label=var)
                plt.savefig(os.path.join(args.outputfile,
                                         '2Dslice_' + "_".join([pair.replace(":","") for pair in args.slice_plot]) 
                                         + "_" + var + "_vs_" + sliced.index.names[0] +"_" + sliced.index.names[1] + ".png"),
                            dpi=150)
                plt.clf()
                plt.close()
        
        
