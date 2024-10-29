import numpy as np
import pandas as pd
import os
import glob
import ctable_tools

Fuel = 'CH4'
filepattern = 'nonpremixed_flames/extinction*csv'
prog_definition = {"H2O":1, "CO2":1, "H2":1, "CO":1}
keep_vars = ["RHO","T","DIFF","VISC","SRC_PROG",
             "Y-H2O","Y-H2","Y-CO","Y-CO2","Y-O2","Y-N2","Y-CH4","Y-OH",
             "SRC_H2O","SRC_H2","SRC_CO","SRC_CO2","SRC_O2","SRC_N2","SRC_CH4","SRC_OH",
             "Y-CH2O","Y-HO2","PROG","lnRHO","invRHO"]
# all SRCs added to use table for network training
outfile = "nonpremixed.ctb"
Zst = 0.0551538
Zgrid = np.concatenate([np.linspace(0,2*Zst,41),
                         np.linspace(2*Zst,0.4,30)[1:],
                         np.linspace(0.4,1.0,31)[1:]])
nC = 100
# revised 
Zgrid = np.concatenate([np.linspace(0,2*Zst,101),
                         np.linspace(2*Zst,0.4,100)[1:],
                         np.linspace(0.4,1.0,101)[1:]])
nC = 300

def compute_prog(data, prog_def, prefix='Y-', suffix=''):
    prog = np.zeros(data.shape[0])
    for spec in prog_def.keys():
        prog += prog_def[spec] * np.array(data[prefix + spec + suffix])
    return prog

# Read in Data Files
files = sorted(glob.glob(filepattern))[::-1] # must be in Lambda (generalized progvar) order
Cmax = 0.0
dfindex = pd.MultiIndex.from_product([files,Zgrid], names = ['file','ZMIX'])
interpdata = pd.DataFrame(index=dfindex, columns=keep_vars, dtype=np.float64)
for filename in files:
    data = pd.read_csv(filename)
    data['lnRHO'] = np.log(data["RHO"])
    data['invRHO'] = 1.0/data["RHO"]
    data['PROG'] = compute_prog(data, prog_definition)
    data['SRC_PROG'] = compute_prog(data, prog_definition, prefix='SRC_')
    Cmax = max(Cmax, np.max(data['PROG']))
    print(filename, Cmax)
    # interpolate onto Zgrid
    for Zval in Zgrid:
        for column in keep_vars:
            interpdata[column][filename] = np.interp(Zgrid, data['Zmix'][::-1], data[column][::-1])
    # set source term to 0 for min and max Lambda
    if filename == files[0] or filename == files[-1]:
        for column in keep_vars:
            if column.startswith('SRC'):
                interpdata[column][filename] = 0.0

Cgrid = np.linspace(0,Cmax,nC)
dfindex = pd.MultiIndex.from_product([Zgrid,Cgrid], names = ['ZMIX','PROG'])
finaldata = pd.DataFrame(index=dfindex, columns=keep_vars, dtype=np.float64, data=0.0)
#interpolate into Cgrid
for Zval in Zgrid:
    for column in keep_vars:
        finaldata[column][Zval] = np.interp(Cgrid, interpdata['PROG'][:,Zval], interpdata[column][:,Zval])

# Save
ctable_tools.write_chemtable_binary(outfile, finaldata, "NONPREMIXED", "Pele")
ctable_tools.print_chemtable(finaldata)

# Verify saved data can be read correctly
newtable, name = ctable_tools.read_chemtable_binary(outfile)
assert finaldata.equals(newtable), "Loaded table does not match saved, saving did not work properly"
