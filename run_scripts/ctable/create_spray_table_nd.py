"""
Create chemtable for nonreacting multicomponent spray evaporation problems.

The table will have a number of dimensions (mixture fractions) equal to the number
of spray components, becuase mixing with oxidizer is also considered. If the
``use_fmix`` option is specified, instead of standard mixture fractions (sum to 1,
state space is the unit simplex), the table is populated in terms of normalized
mixture fractions (each goes from 0 to 1, state space is a unit cube). Liquid
fuel temperatures and latent heats are specified for each fuel; it is assumed
that the gas phase temperature adjacent to the droplet is temperature that conserves
enthalpy (i.e., it is lower than the liquid temperature due to the vaporization
process).

This script uses Cantera and the user must specify a Cantera yaml format chemical
mechanism. The `liquid_fuels_nonreacting` mechanism that comes with PelePhysics
is likely a good choice.

Usage
-----

Invoke on the command line::

    python create_spray_table_nd.py <input_file.toml>

Input File
----------

The input file is a TOML format file with sections ``[phys]`` and ``[table]``,
corresponding to phsyical/BC inputs and parameters for the table. See the
sample input file for more details.
"""

if __name__ == "__main__":

    import cantera as ct
    import numpy as np
    import pandas as pd
    from cmlm import ctable_tools
    from cmlm.utils import TomlParmParse

    # Load inputs
    tpp = TomlParmParse("create_spray_table_nd.toml", allow_cl_override=True)
    print(tpp.data)

    # Create mixing streams
    mechanism = tpp.get("phys", "mechanism")
    T_ox = tpp.get("phys", "T_ox")
    X_ox = tpp.get("phys", "X_ox")
    pressure = tpp.get("phys", "pressure")
    liqTfuel = tpp.get("phys", "liqTfuel")
    X_fuel = tpp.get("phys", "X_fuel")
    deltaHvap = tpp.get("phys", "deltaHvap")
    ox = ct.Solution(mechanism)
    ox.TPX = T_ox, pressure, X_ox
    oxstream = ct.Quantity(ox, constant="HP")
    fuelstreams = []
    Nfuel = len(X_fuel)
    for ii, fcomps in enumerate(X_fuel):
        fu = ct.Solution(mechanism)
        fu.TPX = liqTfuel[ii], pressure, X_fuel[ii]
        # account for enthalpy of vaporization
        fu.HPX = fu.enthalpy_mass - deltaHvap[ii], fu.P, fu.Y
        fuelstreams.append(ct.Quantity(fu, constant="HP"))
        print(
            "Fuel stream {} ({}): liquid T is {} and gaseous T is {}".format(
                ii, X_fuel[ii], liqTfuel[ii], fu.T
            )
        )
    streams = [oxstream] + fuelstreams

    # Create table

    grids = []
    for grid in tpp.get("table", "grid")[:Nfuel]:
        grids.append(np.linspace(0.0, 1.0, grid))
    use_fmix = tpp.get("table", "use_fmix")
    if use_fmix:
        dimnames = ["ZMIX"]
        for ii in range(Nfuel - 1):
            dimnames.append("FMIX" + str(ii))
    else:
        dimnames = ["ZMIX" + str(ii) for ii in range(Nfuel)]

    dfindex = pd.MultiIndex.from_product(grids, names=dimnames)
    dfcols = ["T", "RHO", "DIFF", "WBAR", "VISC", "CP"] + [
        "Y-" + spec.split(":")[0] for spec in X_fuel
    ]
    df = pd.DataFrame(index=dfindex, columns=dfcols, dtype=np.float64)

    for comp in dfindex:
        remainder = 1.0
        comps = np.array(comp)
        if use_fmix:
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
        df.loc[comp] = [
            mixture.T,
            mixture.density,
            mixture.thermal_conductivity / mixture.cp,
            mixture.mean_molecular_weight,
            mixture.viscosity,
            mixture.cp,
        ] + [mixture.Y[mixture.species_index(spec.split(":")[0])] for spec in X_fuel]

    # Meta data generation
    species_list = [spec.split(":")[0] for spec in X_fuel]
    species_idx_list = [mixture.species_index(sp) for sp in species_list]

    mdatfi = tpp.get("table", "metadata_file")
    with open(mdatfi, "w") as fi:
        fi.write("manifold.has_species_mw = true\n")
        for i in range(len(species_list)):
            fi.write(
                "manifold."
                + species_list[i]
                + "_mw = "
                + str(mixture.molecular_weights[species_idx_list[i]])
                + "\n"
            )
        fi.write("manifold.nominal_pressure_cgs = " + str(pressure * 10.0))

    ctable_tools.convert_chemtable_units(df, "mks2cgs")
    df["RHOinv"] = 1.0 / df["RHO"]
    df["lnRHO"] = np.log(np.array(df["RHO"]))
    ctable_tools.print_chemtable(df)
    print(df)

    ofi = tpp.get("table", "filename")
    ctable_tools.write_chemtable_binary(ofi, df, "mixing-only")
