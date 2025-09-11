import cantera as ct
import ctable_tools
import pandas as pd
import toml
import sys
import numpy as np

# ------------------------------------------------------------------------- #
# get_spray_nd_table.py
#
# Simple script to compute an ND chemtable for mixing of N liquid fuel
# components with air, with N mixture fractions corresponding to the N fuels.
#
# Usage:
# python get_spray_nd_table.py <input_file>
# A sample input file "spray_nd.inp" is provided.

# Notes on inputs:
# - The fuel species must exist the provided Cantera mechanism. The
#   `liquid_fuels_nonreacting` mechanism in PelePhysics is a good choice.
# - The user provides a temperature of the liquid for each stream and
#   enthalpy of vaporization, which are used to compute the gas phase
#   temperature corresponding to vaporized liquid.
# - The fuel stream may in principal themselves have multiple components
#   (specified as Cantera format compositions), but mostly it makes sense
#   to treat each component with a separate mixtyre fraction
# ------------------------------------------------------------------------- #


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
dfcols = ["T","RHO","DIFF","WBAR","VISC"]+["Y-"+spec.split(":")[0] for spec in iphys["X_fuel"]]
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
        streams[0].mass = remainder
    nonzeromass = [stream.mass > 0.0 for stream in streams]
    mixture = np.sum(np.array(streams)[nonzeromass])
    df.loc[comp] = ([mixture.T, mixture.density, mixture.thermal_conductivity/mixture.cp, mixture.mean_molecular_weight, mixture.viscosity]
                + [mixture.Y[mixture.species_index(spec.split(":")[0])] for spec in iphys["X_fuel"]])

#Meta data generation
all_species_mwt = []            # List of list, species with molecular weights
species_list = [spec.split(":")[0] for spec in iphys["X_fuel"]]
species_idx_list = [mixture.species_index(sp) for sp in species_list]

for i in range(len(species_list)):    
    row_data = ["manifold."+species_list[i]+"_mwt",mixture.molecular_weights[species_idx_list[i]]]    
    all_species_mwt.append(row_data)

df_meta = pd.DataFrame(all_species_mwt, columns=['SpeciesNamesWithPrefix', 'Molecular_Weight'])
df_meta.to_csv('Manifold_Metadata.txt', sep='=', header=False, index=False, encoding='ascii')


ctable_tools.convert_chemtable_units(df, "mks2cgs")
print(df)
df["RHOinv"] = 1.0/df["RHO"]
df["lnRHO"] = np.log(np.array(df["RHO"]))

ctable_tools.write_chemtable_binary(itable["filename"], df, "mixing-only")
