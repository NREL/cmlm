"""Helper functions for using Cantera."""

import pandas as pd


def save_flame_csv(flame, filename, cp_fuel_species=""):
    """
    Save a csv file with data from a Cantera 1D flame.

    Parameters
    ----------
        flame: cantera 1D flame
            Flame object with data to be saved
        filename: str, path-like
            csv file to save data to
        cp_fuel_species: str
            Save heat capacity for this species in the csv (optional)
    """
    data = pd.DataFrame()
    if hasattr(flame, "mixture_fraction"):
        data["Zmix"] = flame.mixture_fraction(m="N")
    data["X"] = flame.grid
    data["T"] = flame.T
    data["VEL"] = flame.velocity
    data["RHO"] = flame.density_mass
    data["DIFF"] = flame.thermal_conductivity / flame.cp_mass
    data["VISC"] = flame.viscosity
    data["LAMBDA"] = flame.thermal_conductivity
    data["CP"] = flame.cp_mass
    data["MW"] = flame.mean_molecular_weight
    if cp_fuel_species != "":
        sa = flame.to_array()
        sa.TPY = sa.T, sa.P, cp_fuel_species
        data["CP_FUEL"] = sa.cp_mass
    spec_names = [f"Y-{spec}" for spec in flame.gas.species_names]
    spec_y_data = pd.DataFrame(flame.Y.T, columns=spec_names)
    rr_names = [f"SRC_{spec}" for spec in flame.gas.species_names]
    spec_rr_data = pd.DataFrame(
        flame.net_production_rates.T * list(flame.gas.molecular_weights),
        columns=rr_names,
    )
    data = pd.concat([data, spec_y_data, spec_rr_data], axis=1)
    data.to_csv(filename)


def save_table_metadata(gas, filename):
    """
    Save metadata file from Cantera solution object.

    The metadata file will have AMReX ParmParse format and contain necessary data to
    run Spray simulations with tabulated chemistry in PeleLMeX.

    Parameters
    ----------
        gas: Cantera Solution object
            Must have pressure defined
        filename: str, path-like
            Where data will be saved
    """
    with open(filename, "w") as fi:
        fi.write("manifold.has_species_mw = true\n")
        for i, spec in enumerate(gas.species_names):
            fi.write(f"manifold.{spec}_mw = {gas.molecular_weights[i]}\n")
        fi.write(f"manifold.nominal_pressure_cgs = {gas.P * 10.0}\n")  # Convert to CGS
