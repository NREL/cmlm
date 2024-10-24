import cantera as ct
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import ctable_tools as ctable

# Thermo Conditions
press = ct.one_atm
temp = 298.0
phi = 1.0
fuel = 'CH4'
oxid = 'O2:1.0, N2:3.76'
mechanism = 'drm19.yaml'
trans = 'Mix'
progvars = ["CO2","H2O","CO","H2"]
ctable_specs = ["CO2","H2O","CO","H2","N2","O2","OH","CH4","HO2","CH2O"]
outfile_prefix = 'data/prem_drm19_phi1_p1_t298'
net_info_file = None

def load_info_file(filename):
    """Load neural net info file."""

    with open(filename, 'r') as file:
        umap = defaultdict(list)
        line_cont = False
        for line in file:
            try:
                com_idx = line.index("#")
            except:
                com_idx = None
            if com_idx is not None:
                line = line[:com_idx].strip()
                if (not line): continue
            if not line_cont:
                varname, value = line.split(' = ')
            else:
                value = line
            umap[varname.strip()] += value.strip().rstrip("\\").split()
            line_cont = line.strip().endswith("\\")

    dimnames = umap["dimnames"]
    defn0 = "def_" + dimnames[0]
    coeff = np.zeros((len(dimnames), len(umap[defn0])), dtype=np.float32)
    varidx = np.zeros((len(dimnames), len(umap[defn0])), dtype=np.int32)
    srcidx = np.zeros((len(dimnames), len(umap[defn0])), dtype=np.int32)
    combmap = dict()

    for i in range(len(dimnames)):
        defn = "def_" + dimnames[i]
        for j, item in enumerate(umap[defn]):

            c, v = item.split("*")
            c = c.strip()
            v = v.strip()

            coeff[i,j] = float(c)
            varidx[i,j] = umap["varnames"].index(v)
            combmap[v] = j

            if v.startswith("Y-"):
                v = v[2:]
            try:
                idx = umap["varnames"].index("SRC_" + v)
            except ValueError:
                idx = -1
            srcidx[i,j] = idx

    manibiases = np.fromiter(map(float, umap["manibiases"]), dtype=np.float32)
    return umap, coeff, varidx, srcidx, manibiases, combmap

# Read info file if one is provided
if net_info_file is not None:
    umap, coeff, varidx, srcidx, manibiases, combmap = load_info_file(net_info_file)
else:
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
data["lnRHO"] = np.log(data["RHO"])
specRRdata *= 1.0e-3

# Make the chemtable
chemtable = data.drop(columns=(["VEL","X","PROG"]+umap["dimnames"]))
for spec in ctable_specs:
    chemtable["Y-"+spec] = specYdata[spec]
chemtable = pd.DataFrame(chemtable.values,
                         columns = chemtable.columns,
                         index = pd.MultiIndex.from_product([data["PROG"]],names=['PROG']))
ctable.write_chemtable_binary(outfile_prefix+'.ctb', chemtable, "1DFGM")
ctable.print_chemtable(chemtable)

chemtable.drop(columns=["SRC_PROG"], inplace=True)
ctable.write_chemtable_binary(outfile_prefix+'_norxn.ctb', chemtable, "1DFGM")
ctable.print_chemtable(chemtable)

for spec in ctable_specs:
    chemtable["SRC_"+spec] = np.array(specRRdata[spec])
ctable.write_chemtable_binary(outfile_prefix+'_allrxn.ctb', chemtable, "1DFGM")
ctable.print_chemtable(chemtable)

# Make the PMF dat files: Detailed Chem and Manifold
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
df2 = pd.DataFrame(data[["PROG","RHO"]].values, columns=['X0','XRHO'], index= data.index)
write_dat_file(outfile_prefix+'.dat', pd.concat([df,specXdata],axis=1))
write_dat_file(outfile_prefix+'_mani.dat', pd.concat([df,df2],axis=1))

if net_info_file is not None:
    df3 = pd.DataFrame(data[umap["dimnames"]+["RHO"]].values + manibiases,
                       columns=(["X{}".format(i) for i in range(len(umap["dimnames"]))]+['XRHO']),
                       index= data.index)
    write_dat_file(outfile_prefix+'_mani_nn.dat', pd.concat([df,df3],axis=1))
