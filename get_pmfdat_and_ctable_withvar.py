import cantera as ct
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import ctable_tools as ctable
from scipy.stats import beta, norm
import matplotlib.pyplot as plt

# Thermo Conditions
press = ct.one_atm
temp = 298.0
phi = 1.0
fuel = 'CH4'
oxid = 'O2:1.0, N2:3.76'
mechanism = '~/Software/PeleLMeX/Submodules/PelePhysics/Mechanisms/drm19/mechanism.yaml'
trans = 'Mix'
progvars = ["CO2","H2O","CO","H2"]
ctable_specs = ["CO2","H2O","CO","H2","N2","O2","OH","CH4","HO2","CH2O"]
outfile_prefix = 'data/prem_drm19_phi1_p1_t298'
PROGvals = np.linspace(0,1,101)
PROGVARvals = np.linspace(0,0.25,41)
filter_width = 0.01875 # cm

coeff, varidx, srcidx, manibiases, combmap = (None,)*5
umap = defaultdict(list)

# Flame Numerics
width = 0.1
loglevel = 1
ratio = 2
slope = 0.025
curve = 0.025
prune = 0.01
max_points = 10000

# Set up the flame
gas = ct.Solution(mechanism)
gas.set_equivalence_ratio(phi, fuel, oxid)
gas.TP = temp, press
flame = ct.FreeFlame(gas, width=width)
flame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
flame.set_max_grid_points(1,max_points)

flame.transport_model = trans

# Solve Flame
flame.solve()

# Get all desired data
data = pd.DataFrame()
data["X"] = flame.grid
data["T"] = flame.T
data["VEL"] = flame.velocity
data["RHO"] = flame.density_mass
data["DIFF"] = flame.thermal_conductivity / flame.cp_mass
data["VISC"] = flame.viscosity

specXdata = pd.DataFrame(flame.X.T,
                         columns=gas.species_names)
specYdata = pd.DataFrame(flame.Y.T,
                         columns=gas.species_names)

rxnrates = flame.net_production_rates.T * list(gas.molecular_weights)
specRRdata = pd.DataFrame(rxnrates,columns=gas.species_names)

# Compute prgress variable and its source
# ensure prog is monotonic
# source term 0 at min and max to ensure boundedness
data["PROG"] = specYdata[progvars].sum(axis=1)
prog = np.array(data["PROG"])
# Progress variable should increase monotonically (small deviations may arise due to numerics and will be ignored)
assert min(np.diff(data.index)) > -1e-10
dropvals = []
for ii in data.index[1:]:
    if data['PROG'][ii] <= max(data['PROG'][:ii]):
        dropvals.append(data.index[ii])
data.drop(index=dropvals, inplace=True)
specRRdata.drop(index=dropvals, inplace=True)
specYdata.drop(index=dropvals, inplace=True)
specXdata.drop(index=dropvals, inplace=True)

data["SRC_PROG"] = specRRdata[progvars].sum(axis=1)
data.loc[0,"SRC_PROG"] = 0.0
data.loc[len(data.index)-1,"SRC_PROG"] = 0.0

# Renormalize progress variable (0 - unburned, 1 = equilibrium)
data["PROG"].iloc[0] = 0.0;
normval = 1/data['PROG'].iloc[-1];
data["PROG"] *= normval;
data["SRC_PROG"] *= normval;

# Compute manifold variable(s) for net
for i, dim in enumerate(umap["dimnames"]):
    data[dim] = 0.0
    for k in combmap:
        if k.startswith('Y-'):
            k_slc = k[2:]
        else:
            k_slc = k
        data[dim] += coeff[i, combmap[k]] * specYdata[k_slc]

# Convert to CGS units
ctable.convert_chemtable_units(data)
data["PROG*SRC_PROG"] = data["PROG"]*data["SRC_PROG"]
specRRdata *= 1.0e-3

# Create structure for convoluted data
ctabledata = data.drop(columns=(["VEL","X",]+umap["dimnames"]))
for spec in ctable_specs:
    ctabledata["Y-"+spec] = specYdata[spec]
dfindex = pd.MultiIndex.from_product([PROGvals, PROGVARvals], names = ['PROG','PROGVAR'])
interpdata = pd.DataFrame(index=dfindex, columns=ctabledata.columns, dtype=np.float64)

# Preliminaries for non-density-weighted variables
ctabledata['RHO'] = 1.0/ctabledata['RHO']
ctabledata["PROG*SRC_PROG"] = ctabledata["PROG*SRC_PROG"] * ctabledata['RHO']
ctabledata["SRC_PROG"] = ctabledata["SRC_PROG"] * ctabledata['RHO']

# set up data for convolution
ctabledata = ctabledata.T
Carr = np.array(data['PROG'])
CMidPoints = [0.0] + list(0.5*(Carr[1:] + Carr[:-1])) + [1.0]
lastcol = ctabledata.columns[-1]

# convolute with beta distribution
for Cval in PROGvals:
    for Cvar in PROGVARvals:
        # 0 variance: interpolate
        if Cvar == 0.0:
            for column in interpdata.columns:
                interpdata[column][Cval, Cvar] = np.interp(Cval, Carr, ctabledata.loc[column])

        # more than max variance:
        elif Cval == 0.0 or Cval == 1.0 or Cvar/(Cval*(1-Cval)) >= 1.0:
            a = 1.0 - Cval
            b = Cval
            interpdata.loc[Cval, Cvar] = a*ctabledata[0] + b*ctabledata[lastcol]

        # variance within realizable range:
        else :
            a = -Cval*(Cval*Cval - Cval + Cvar) / Cvar
            b = -(1-Cval)*(Cval*Cval - Cval + Cvar) / Cvar
            betadist = beta(a, b)
            cdf = betadist.cdf(CMidPoints)
            pdf = cdf[1:] - cdf[:-1]
            interpdata.loc[Cval,Cvar] = np.matmul(ctabledata.values, pdf)

# post filtering conversions for non-density-weighted variables
interpdata['RHO'] = 1.0/interpdata['RHO']
interpdata["PROG*SRC_PROG"] = interpdata["PROG*SRC_PROG"] * interpdata['RHO']
interpdata["SRC_PROG"] = interpdata["SRC_PROG"] * interpdata['RHO']

# Add in some other useful vriables
interpdata["SRC_PROGVAR"] = 2.0*(interpdata["PROG*SRC_PROG"] - (interpdata["PROG"] * interpdata["SRC_PROG"]))
interpdata["lnRHO"] = np.log(interpdata["RHO"])

# Make the chemtable
ctable.write_chemtable_binary(outfile_prefix+'.ctb', interpdata, "2DFGM")
ctable.print_chemtable(interpdata)

# Make the PMF dat files: Detailed Chem and Manifold (and filtered Manifold)
def write_dat_file(fname, df):
    with open(fname,'w') as fi:
        line1 = "".join(["VARIABLES ="]
                        + [' "{}"'.format(var.split(' ')[0]) for var in df.columns[:4]]
                        + [' "{}"'.format(var.upper()) for var in df.columns[4:]])
        fi.write(line1+'\n')
        line2 = " ZONE I={} FORMAT=POINT\n".format(df.shape[1])
        fi.write(line2)
        print('Reformated file has these variables:')
        print(line1)
        for idex, row in df.iterrows():
            linen = "".join(['{:<26.15g}'.format(x) for x in row])+'\n'
            fi.write(linen)

rename = {'VEL':'u', 'T':'temp','RHO':'rho'}
keepvars = ["X","T","VEL","RHO"]
df = data[keepvars].rename(columns=rename)
data["PROGVAR"] = 0.0
df2 = pd.DataFrame(data[["PROGVAR","PROG","RHO"]].values, columns=['PROGVAR','PROG','XRHO'], index=data.index)
write_dat_file(outfile_prefix+'.dat', pd.concat([df,specXdata],axis=1))
manidf = pd.concat([df,df2],axis=1)
write_dat_file(outfile_prefix+'_mani.dat', manidf)

# filtered version
manidf["PROGVAR"] = manidf['PROG'] * manidf['PROG'] # acutally PROG2 for now

# start by desnity weighting the favre filtered variables
favrevariables = ['temp','u','PROGVAR','PROG']
for var in favrevariables:
    manidf[var] *= manidf['rho']

# do the filtering
def gaussian(x, mean, stdev):
    ''' Compute gaussian filter kernel at points x with specified
    location (mean) and width (stdev). '''
    # gauss = np.exp(-0.5 * (x - mean) / stdev**2)
    midpoints = x[:-1] + 0.5*np.diff(x)
    cdf = norm.cdf(midpoints, scale=stdev, loc=mean)
    gauss = np.zeros(len(x))
    gauss[0] = cdf[0]
    gauss[1:-1] = np.diff(cdf)
    gauss[-1] = 1 - cdf[-1]
    return gauss

filtered = pd.DataFrame(index=manidf.index, columns=manidf.columns, dtype=np.float64, data=0.0)
for idx, xmean in enumerate(manidf['X']):
    filter_weights = gaussian(manidf['X'], xmean, filter_width)
    filtered.iloc[idx,:] = np.matmul(manidf.to_numpy().T, filter_weights)

filtered['X'] = manidf['X']

# divide by desnity to get favre filtered variables
for var in favrevariables:
    filtered[var] /= filtered['rho']
filtered['PROGVAR'] = filtered['PROGVAR'] - filtered['PROG'] * filtered['PROG']

write_dat_file(outfile_prefix+'_mani_filter{0:.3e}.dat'.format(filter_width), filtered)
