import numpy as np
import pandas as pd
import os
import glob
import ctable_tools
import time
from scipy.stats import beta, norm

# ---------------------------------------------------------#
#                             Inputs                       #
# ---------------------------------------------------------#

verbose = 1
use_MPI = True
filepattern = 'nonpremixed_flames/extinction*csv'
prog_definition = {"H2O":1, "CO2":1, "H2":1, "CO":1}
keep_vars = ["RHO","T","DIFF","VISC",
             "Y-H2O","Y-H2","Y-CO","Y-CO2","Y-O2","Y-N2","Y-CH4","Y-OH",
             "SRC_H2O","SRC_H2","SRC_CO","SRC_CO2","SRC_O2","SRC_N2","SRC_CH4","SRC_OH",
             "Y-CH2O","Y-HO2"]
outfile = "nonpremixed.ctb"
Zst = 0.0551538

Zgrid = np.concatenate([np.linspace(0,2*Zst,41),
                         np.linspace(2*Zst,0.4,30)[1:],
                         np.linspace(0.4,1.0,31)[1:]])
Zvargrid = np.linspace(0,0.25,51)
nC = 100

Zgrid = np.concatenate([np.linspace(0,2*Zst,61),
                         np.linspace(2*Zst,0.4,50)[1:],
                         np.linspace(0.4,1.0,51)[1:]])
Zvargrid = np.linspace(0,0.25,100)
nC = 100


#Zgrid = np.concatenate([np.linspace(0,2*Zst,21),
#                         np.linspace(2*Zst,0.4,20)[1:],
#                         np.linspace(0.4,1.0,21)[1:]])
#Zvargrid = np.linspace(0,0.25,26)
#nC = 30

# ---------------------------------------------------------#
#                    Start of Main                         #
# ---------------------------------------------------------#
def compute_prog(data, prog_def, prefix='Y-', suffix=''):
    prog = np.zeros(data.shape[0])
    for spec in prog_def.keys():
        prog += prog_def[spec] * np.array(data[prefix + spec + suffix])
    return prog

# Read in Data Files
files = sorted(glob.glob(filepattern))[::-1] # must be in increasing Lambda (generalized progvar) order

if use_MPI:
    from  mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
else:
    rank = 0
    nprocs = 1
    
Cmax = 0.0
dfindex = pd.MultiIndex.from_product([Zgrid,Zvargrid], names = ['ZMIX','ZMIXVAR'])
data = pd.read_csv(files[0])
for var in keep_vars:
    if var not in data.columns:
        raise RuntimeError("Requested variable {} not found in data file".format(var))
keep_cols = [column for column in data.columns if column in keep_vars] + ["PROG", "SRC_PROG"]
interpdata = {}

if (verbose > 0 and rank == 0):
    print("Loading and convoluting flamelets...", flush=True)
    
for ii in range(len(files))[rank::nprocs]:
    filename = files[ii]
    interpdata[ii] = pd.DataFrame(index=dfindex, columns=keep_cols, dtype=np.float64)
    tstart = time.time()
    data = pd.read_csv(filename)[::-1] # reverse order so it goes in ascending Z order
    data['PROG'] = compute_prog(data, prog_definition)
    data['SRC_PROG'] = compute_prog(data, prog_definition, prefix='SRC_')
    Cmax = max(Cmax, np.max(data['PROG']))
    tread = time.time() - tstart
    if (verbose > 0):
        print("Rank {} reading file {} ({}/{}): CMax = {}, NZ = {}".format(rank, filename, ii+1, len(files), Cmax, data.shape[0]), flush=True)
    
    # Convolute with Beta PDF based on Z,Zvar
    # first get non-density-weighted variables ready for density-weighted PDF convolution
    for col in keep_cols:
        if col.startswith("SRC_"):
            data[col] = data[col] / data['RHO']
    data['RHO'] = 1.0/data['RHO']
    
    # set up data for convolution - ensure Z is monotonic, remove redundant values
    dropindex = []
    Zprev = -1e10
    for val in data.index:
        if data.loc[val,'Zmix'] == Zprev:
            dropindex.append(val)
        assert(data.loc[val,'Zmix'] >= Zprev)
        Zprev = data.loc[val,'Zmix']
    data.drop(index=dropindex, inplace=True)
    Zin = np.array(data['Zmix'])
    assert(Zin[0] == 0.0 and Zin[-1] == 1.0)
    data = data[keep_cols]
    data = data.T
    ZinMidPoints = [0.0] + list(0.5*(Zin[1:] + Zin[:-1])) + [1.0]
    lastcol = data.columns[-1]
    firstcol = data.columns[0]
    
    # now do the convolution for each Z,Zvar
    for Zval in Zgrid:
        for Zvar in Zvargrid:
            # 0 variance: interpolate
            if Zvar == 0.0:
                for column in interpdata[ii].columns:
                    interpdata[ii].loc[(Zval, Zvar), column] = np.interp(Zval, Zin, data.loc[column])
                
            # more than max variance - double delta at C=0 and C=1
            elif Zval == 0.0 or Zval == 1.0 or Zvar/(Zval*(1-Zval)) >= 1.0:
                a = 1.0 - Zval
                b = Zval
                interpdata[ii].loc[Zval, Zvar] = a*data[firstcol] + b*data[lastcol]
                
            # variance within realizable range - convolute with Beta
            else:
                a = -Zval*(Zval*Zval - Zval + Zvar) / Zvar
                b = -(1-Zval)*(Zval*Zval - Zval + Zvar) / Zvar
                betadist = beta(a, b)
                cdf = betadist.cdf(ZinMidPoints)
                pdf = cdf[1:] - cdf[:-1]
                #pdf = np.ones(data.shape[1])
                interpdata[ii].loc[Zval, Zvar] = np.matmul(data.values, pdf)

    tconv = time.time() - tstart - tread
    
    # After convolution - restore non-density weighted variables
    interpdata[ii]['RHO'] = 1.0/interpdata[ii]['RHO']
    for col in keep_cols:
        if col.startswith("SRC_"):
            interpdata[ii][col] = interpdata[ii][col] * interpdata[ii]["RHO"]
                
    # set source term to 0 for min and max Lambda
    if filename == files[0] or filename == files[-1]:
        for column in keep_cols:
            if column.startswith('SRC'):
                interpdata[ii][column] = 0.0

    tend = time.time() - tstart - tread - tconv
    if (verbose > 1) :
        print("timing - read: {}, conv: {}, finalize: {}".format(tread, tconv, tend))

# Communicate and concatenate data
if (verbose > 0 and rank == 0):
    print("Communicating and concatenating data for interpolation...", flush=True)

globalcmax = comm.reduce(Cmax, op=MPI.MAX, root=0)
gathereddata = comm.gather(interpdata, 0)

# only do the rest of the work on a single rank
if rank == 0:
    interpdata = pd.concat([pd.concat(gdat, names=['file']+interpdata[0].index.names) for gdat in gathereddata])
    interpdata = interpdata.sort_index()
    interpdata = interpdata.to_numpy().reshape(len(files), len(Zgrid), len(Zvargrid), len(keep_cols))
    interpdata = np.moveaxis(interpdata,0,2) # order now (Z, Zvar, file, Variable)

    Cmax = globalcmax
    Cgrid = np.linspace(0,Cmax,nC)
    dfindex = pd.MultiIndex.from_product([Zgrid,Zvargrid,Cgrid], names = ['ZMIX','ZMIXVAR','PROG'])
    finaldata = np.zeros((len(Zgrid), len(Zvargrid), nC, len(keep_cols)))
    idxC = keep_cols.index('PROG')

    # interpolate into Cgrid
    if (verbose > 0 and rank == 0):
        print("Interpolating onto C grid...", flush=True)
        tstart = time.time()
    for ii,Zval in enumerate(Zgrid):
        for jj,Zvar in enumerate(Zvargrid):
            for icol in range(len(keep_cols)):
                finaldata[ii,jj,:,icol]= np.interp(Cgrid, interpdata[ii,jj,:,idxC], interpdata[ii,jj,:,icol])
                    
    if(verbose > 1 and rank == 0): 
        print(" done interpolating in {} seconds".format(time.time() - tstart), flush=True)

    finaldata = pd.DataFrame(index=dfindex, columns=keep_cols, dtype=np.float64, data=finaldata.reshape((len(Zgrid)*len(Zvargrid)*nC, len(keep_cols))))
    
    if (finaldata.isnull().any().any()) :
        raise RuntimeError("found NaN in table, stopping")
    
    # add some extra variables
    finaldata['logRHO'] = np.log(finaldata['RHO'])
    finaldata['RHOinv'] = 1.0 / finaldata['RHO']
    # No unit conversion -> assume data in incoming flamelets was already converted

    # Save
    if (verbose > 0 and rank == 0):
        print("Writing and printing chemtable...")
    ctable_tools.write_chemtable_binary(outfile, finaldata, "NONPREMIXED", "Pele")
    ctable_tools.print_chemtable(finaldata)

    # Verify saved data can be read correctly
    newtable, name = ctable_tools.read_chemtable_binary(outfile)
    assert finaldata.equals(newtable), "Loaded table does not match saved, saving did not work properly"

    if (verbose > 0 and rank == 0):
        print("Done. Table verified.")
