import numpy as np
import pandas as pd
import os
import glob
import ctable_tools

Fuel = 'CH4'
filepattern = 'autoignition/a*.csv'
prog_definition = {"H2O":1, "CO2":1, "H2":1, "CO":1}
keep_vars = ["RHO","T","DIFF","VISC","SRC_PROG",
             "Y-H2O","Y-H2","Y-CO","Y-CO2","Y-O2","Y-N2","Y-CH4","Y-OH",
             "Y-CH2O","Y-HO2"]
outfile = "autoignition.ctb"

def compute_prog(data, prog_def, prefix='Y-', suffix=''):
    prog = np.zeros(data.shape[0])
    for spec in prog_def.keys():
        prog += prog_def[spec] * np.array(data[prefix + spec + suffix])
    return prog

# Read in Data Files
files = sorted(glob.glob(filepattern)) # must be in Z order
alldata = []
Zvals = []
Cmax = []
for filename in files:
    data = pd.read_csv(filename)
    data.index = compute_prog(data, prog_definition)
    data.index.name = 'PROG'
    
    # Progress variable should increase monotonically (small deviations may arise due to numerics and will be ignored)
    assert min(np.diff(data.index)) > -1e-10
    dropvals = []
    for ii in range(1,len(data.index)):
        if data.index[ii] <= max(data.index[:ii]):
            dropvals.append(data.index[ii])
    data.drop(dropvals)
    
    # Add source terms to data
    data['SRC_PROG'] = compute_prog(data, prog_definition, prefix='SRC_')
    data['SRC_PROG'].iloc[-1] = 0.0
    alldata.append(data[keep_vars])
    
    Zvals.append(data['Y-'+Fuel].iloc[0])
    Cmax.append(np.max(data.index))
    
# Interpolate onto grid
assert min(np.diff(Zvals)) > 0 # Zvals must increase monotonically
Cgrid = np.array(alldata[np.argmax(Cmax)].index) 
dfindex = pd.MultiIndex.from_product([Zvals, Cgrid], names = ['ZMIX','PROG'])
interpdata = pd.DataFrame(index=dfindex, columns=keep_vars)
for Zval,data in zip(Zvals,alldata):
    for column in keep_vars:
        interpdata[column][Zval] = np.interp(Cgrid, data.index, data[column])

# Save
ctable_tools.write_chemtable_binary(outfile, interpdata, "AUTOIGNITION")
ctable_tools.print_chemtable(interpdata)
