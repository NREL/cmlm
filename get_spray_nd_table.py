import cantera as ct
import ctable_tools
import pandas as pd
import toml
import sys
import numpy as np

# Load inputs
if len(sys.argv) != 2:
    raise RuntimeError("Invalid Usage: input file is single command line argument")
infile = sys.argv[1]
try:
    with open(infile) as tomlfile:
       inputs = toml.load(tomlfile)
except FileNotFoundError:
    raise FileNotFoundError("Input file < " + sys.argv[1] + " > not found!")
iphys = inputs["phys"]
itable = inputs["table"]

# Create mixing streams
ox = ct.Solution(iphys["mechanism"])
ox.TPX = iphys["T_ox"], iphys["pressure"], iphys["X_ox"]
oxstream = ct.Quantity(ox, constant="HP")
fuelstreams = []
Nfuel = len(iphys["X_fuel"])
for ii, fcomps in enumerate(iphys["X_fuel"]):
    fu = ct.Solution(iphys["mechanism"])
    fu.TPX = iphys["liqTfuel"][ii], iphys["pressure"], iphys["X_fuel"][ii]
    # account for enthalpy of vaporization
    fu.HPX = fu.enthalpy_mass - iphys["deltaHvap"][ii], fu.P, fu.Y
    fuelstreams.append(ct.Quantity(fu,constant="HP"))
    print("Fuel stream {} ({}): liquid T is {} and gaseous T is {}".format(ii, iphys["X_fuel"][ii], iphys["liqTfuel"][ii], fu.T))
streams = [oxstream] + fuelstreams

# Create table
grids =[]
for grid in itable["grid"][:Nfuel]:
    grids.append(np.linspace(0.0,1.0,grid))
if itable["use_fmix"]:
    dimnames = ["ZMIX"]
    for ii in range(Nfuel - 1):
        dimnames.append("FMIX"+str(ii))
else:
    dimnames = ["ZMIX"+str(ii) for ii in range(Nfuel)]

dfindex = pd.MultiIndex.from_product(grids, names=dimnames)
dfcols = ["T","RHO","DIFF","VISC"]+["Y-"+spec.split(":")[0] for spec in iphys["X_fuel"]]
df = pd.DataFrame(index=dfindex, columns=dfcols, dtype=np.float64)

for comp in dfindex:
    remainder = 1.0
    comps = np.array(comp)
    if itable["use_fmix"]:
        comps[0] = 1 - comps[0]
        for ii, stream in enumerate(streams[:-1]):
            stream.mass = remainder * comps[ii]
            remainder = remainder * (1 - comps[ii])
        streams[-1].mass = remainder
    else:
        for ii, stream in enumerate(streams[1:]):
            strm_mass = min(remainder, comps[ii])
            remainder = remainder - strm_mass
            stream.mass = strm_mass
            print(ii, comps, remainder)
        streams[0].mass = remainder
    nonzeromass = [stream.mass > 0.0 for stream in streams]
    mixture = np.sum(np.array(streams)[nonzeromass])
    df.loc[comp] = ([mixture.T, mixture.density, mixture.thermal_conductivity/mixture.cp, mixture.viscosity]
                + [mixture.Y[mixture.species_index(spec.split(":")[0])] for spec in iphys["X_fuel"]])
    print (df.loc[comp]["RHO"])

ctable_tools.convert_chemtable_units(df, "mks2cgs")
print(df)
df["RHOinv"] = 1.0/df["RHO"]
df["lnRHO"] = np.log(np.array(df["RHO"]))

ctable_tools.write_chemtable_binary(itable["filename"], df, "mixing-only")
