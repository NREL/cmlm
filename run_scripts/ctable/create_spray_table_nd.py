"""
Create chemtable for nonreacting multicomponent spray evaporation problems.

The table will have a number of dimensions (mixture fractions) equal to the number
of spray components, because mixing with oxidizer is also considered. If the
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

Uses TOML format input files (see example input file for more details)::

    python create_spray_table_nd.py create_spray_table_nd.toml
"""

if __name__ == "__main__":

    import cantera as ct
    import numpy as np
    import pandas as pd

    from cmlm import ctable_tools
    from cmlm.utils import TomlParmParse

    # Load inputs
    pp = TomlParmParse.parse_args(
        description="Create chemtable for nonreacting"
        " multicomponent spray evaporation problems."
    )

    # Create mixing streams

    ppp = pp["phys"].doc("Physical conditions/models/parameters")
    mechanism = ppp.get("mechanism", doc="path to mechanism file (yaml)")
    T_ox = ppp.get("T_ox", doc="ambient temp, K")
    X_ox = ppp.get("X_ox", doc="Cantera composition string")
    pressure = ppp.get("pressure", doc="ambient pressure, Pa")
    liq_temp_fuel = ppp.get("liq_temp_fuel", doc="liquid temps for each fuel, K")
    X_fuel = ppp.get("X_fuel", doc="Cantera composition string")
    fuel_species_list = [spec.split(":")[0] for spec in X_fuel]
    species_list = ["O2"] + fuel_species_list
    delta_h_vap = ppp.get("delta_h_vap", doc="Latent heats for each fuel, J/kg")

    ox = ct.Solution(mechanism)
    ox.TPX = T_ox, pressure, X_ox
    oxstream = ct.Quantity(ox, constant="HP")
    fuelstreams = []
    Nfuel = len(X_fuel)
    for ii in range(Nfuel):
        fu = ct.Solution(mechanism)
        fu.TPX = liq_temp_fuel[ii], pressure, X_fuel[ii]

        # note fuel stream does not yet account for enthalpy of vaporization
        # here we do a test just to see what the temperature will be
        fu_vap = ct.Solution(mechanism)
        T_min = ppp.get("T_min", doc="min temperature allowed in gas phase, K")
        try:
            fu_vap.HPY = fu.enthalpy_mass - delta_h_vap[ii], fu.P, fu.Y
            print(
                f"Fuel stream {ii} ({X_fuel[ii]}): liquid T is"
                + f" {liq_temp_fuel[ii]} and gaseous T is {fu_vap.T}"
            )
            if fu_vap.T < T_min:
                print(f"WARNING: T will be limited to T_min = {T_min}")
        except ct.CanteraError:
            print(
                f"WARNING: Negative T due to fuel vaporization"
                f" for fuel stream {ii}, will limit to T_min = {T_min}"
            )
        fuelstreams.append(ct.Quantity(fu, constant="HP"))

    streams = [oxstream] + fuelstreams

    # Create table
    ppt = pp["table"].doc("Table setup inputs")
    grids = []
    grid_sizes = ppt.get(
        "grid", doc="number of grid points for each table dimension (length Nfuels)"
    )
    for grid in grid_sizes[:Nfuel]:
        grids.append(np.linspace(0.0, 1.0, grid))
    use_fmix = ppt.get(
        "use_fmix",
        default=False,
        doc="Tabulate in terms of fuel premixing fractions rather than mixture fractions",
    )
    if use_fmix:
        dimnames = ["ZMIX"]
        for ii in range(Nfuel - 1):
            dimnames.append("FMIX" + str(ii))
    else:
        dimnames = ["ZMIX" + str(ii) for ii in range(Nfuel)]

    dfindex = pd.MultiIndex.from_product(grids, names=dimnames)
    dfcols = ["T", "RHO", "DIFF", "WBAR", "VISC", "CP", "CP_fuel"] + [
        f"Y-{spec}" for spec in species_list
    ]
    df = pd.DataFrame(index=dfindex, columns=dfcols, dtype=np.float64)

    n_limited = 0
    gas = ct.Solution(mechanism)
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

        # Now account for enthalpy of vaporization for real
        enth = mixture.mass * mixture.enthalpy_mass
        for stream, delta_h_i in zip(streams[1:], delta_h_vap):
            enth -= stream.mass * delta_h_i
        try:
            mixture.HP = enth / mixture.mass, mixture.P
            if mixture.T < T_min:
                mixture.TP = T_min, mixture.P
                n_limited += 1
        except ct.CanteraError:
            mixture.TP = T_min, mixture.P
            n_limited += 1

        eps = 1e-10
        gas.TPY = (
            mixture.T,
            mixture.P,
            ", ".join(
                [
                    f"{spec}:{max(mixture.Y[mixture.species_index(spec)],eps)}"
                    for spec in fuel_species_list
                ]
            ),
        )
        cp_fuel = gas.cp

        df.loc[comp] = [
            mixture.T,
            mixture.density,
            mixture.thermal_conductivity / mixture.cp,
            mixture.mean_molecular_weight,
            mixture.viscosity,
            mixture.cp,
            cp_fuel,
        ] + [mixture.Y[mixture.species_index(spec)] for spec in species_list]

    if n_limited > 0:
        print(f"WARNING: {n_limited}/{df.shape[0]} points in table had T limited")

    # Meta data generation
    species_idx_list = [mixture.species_index(sp) for sp in species_list]

    mdatfi = ppt.get(
        "metadata_file", doc="output file path/name for the table metadata"
    )
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

    ofi = ppt.get("filename", doc="output file path/name for the table")
    ctable_tools.write_chemtable_binary(ofi, df, "mixing-only")
