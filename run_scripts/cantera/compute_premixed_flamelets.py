"""
Compute 1D unstrained premixed flamelets using Cantera.

Flames can be computed across a range of conditions including
pressures, temperatures, and compositions. Three methods are
possible for specifying composition sweeps: equivalence ratio,
mixture fraction, and directly specifying a composition.

Uses TOML format input files (see example input file for more details)
and requires specification of an output directory on the command line::

   python compute_premixed_flamelets.py compute_premixed_flamelets.toml -o output
"""

if __name__ == "__main__":

    import itertools
    import os

    import numpy as np
    import pandas as pd
    from cmlm.utils import TomlParmParse
    from cmlm.utils.input_file import scalar_to_list

    import cantera as ct

    # ------------------ Parse relevant inputs ----------------------------#
    pp = TomlParmParse.parse_args(
        description="A tool to compute planar premixed flames using Cantera",
        require_output=True,
    )
    outdir = pp.output_dir

    # Conditions
    ppc = pp["conditions"].doc("Physical conditions for flames")
    pressures = scalar_to_list(ppc.get("pressures", doc="Single pressure or list, atm"))

    composition_type = ppc.get(
        "composition_type",
        choices=["zmixs", "phis", "single"],
        doc="Method of specifying compositions to run",
    )
    if composition_type == "zmixs":
        fuel_comp = ppc.get(
            "fuel_comp", doc="Fuel stream cantera composition, mass basis"
        )
        fuel_temp = ppc.get("fuel_temp", doc="K")
        oxid_comp = ppc.get(
            "oxid_comp", doc="Oxidizer stream cantera composition, mass basis"
        )
        oxid_temp = ppc.get("oxid_temp", doc="K")
        Zvalues = scalar_to_list(ppc.get("Zvalues", doc="Single Z value or list"))
        cond_labels = ["p{:.4f}", "zmix{:.4f}"] # noqa : FS003
        cond_iterator_global = list(itertools.product(pressures, Zvalues))
    elif composition_type == "phis":
        fuel_comp = ppc.get(
            "fuel_comp", doc="Fuel stream cantera composition, mass basis"
        )
        oxid_comp = ppc.get(
            "oxid_comp", doc="Oxidizer stream cantera composition, mass basis"
        )
        temperatures = scalar_to_list(
            ppc.get("temperatures", doc="Single temperature or list, K")
        )
        phis = scalar_to_list(
            ppc.get("phis", doc="Single equivalence ratio or list, K")
        )
        cond_labels = ["p{:.4f}", "T{:.1f}", "phi{:.4f}"] # noqa : FS003
        cond_iterator_global = list(itertools.product(pressures, temperatures, phis))
    else:
        comp = ppc.get("comp", doc="cantera composition, mass basis")
        temperatures = scalar_to_list(
            ppc.get("temperatures", doc="Single temperature or list, K")
        )
        cond_labels = ["p{:.4f}", "T{:.1f}"] # noqa : FS003
        cond_iterator_global = list(itertools.product(pressures, temperatures))
    flame_width = ppc.get(
        "dom_width", default=0.1, doc="Domain width for flame simulations (m)"
    )

    # Models
    ppm = pp["models"].doc("Physical models used by solver")
    mechanism = ppm.get(
        "mechanism", doc="Chemical mechanism to use (Cantera yaml format)"
    )
    transport = ppm.get(
        "transport",
        default="mixture-averaged",
        choices=[
            "mixture-averaged",
            "unity-Lewis-number",
            "multicomponent",
            "high-pressure",
            "high-pressure-Chung",
        ],
        doc="Transport property model from Cantera to use, "
        "see Cantera documentation for more information",
    )
    eos = ppm.get(
        "eos",
        default="",
        doc="Equation of State, must be a phase option in your mechanism yaml file. "
        "If unspecified, the default phase from the mechanism file is used",
    )

    # Flame numerics
    ppf = pp["numerics"].doc(
        "Numerics for nonpremixed flame solve, "
        "see Cantera documentation for more information"
    )
    loglevel = ppf.get("loglevel", default=0)
    ratio = ppf.get("ratio", default=2)
    slope = ppf.get("slope", default=0.2)
    curve = ppf.get("curve", default=0.2)
    prune = ppf.get("prune", default=0.1)
    max_points = ppf.get("max_points", default=10000)

    # Options
    ppo = pp["options"].doc("Additional options for script")
    use_mpi = ppo.get("use_mpi", default=False, doc="Using MPI requires mpi4py")
    if use_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    else:
        rank = 0
        nprocs = 1
    metadata_file = ppo.get(
        "metadata_file",
        default="",
        doc="If specified, save chemtable metadata to this file",
    )
    cp_fuel_species = ppo.get(
        "cp_fuel_species",
        default="",
        doc="If specified, Cantera composition for fuel to compute cp_fuel in "
        "flame output, or 'fuel_comp' to use that input",
    )
    if cp_fuel_species == "fuel_comp":
        cp_fuel_species = fuel_comp

    # ------------------ Solve Flames ---------------------------- #
    cond_iterator = cond_iterator_global[rank::nprocs]
    gas = ct.Solution(mechanism, eos)

    output = pd.DataFrame(
        data=np.nan,
        index=pd.MultiIndex.from_tuples(
            cond_iterator, names=[lab[: lab.index("{")] for lab in cond_labels]
        ),
        columns=["s_L", "T_ad", "l_f", "Nx", "dx_min"],
    )
    for cond in cond_iterator:
        label = "_".join(condlabel.format(c) for condlabel, c in zip(cond_labels, cond))
        print(f"Rank {rank} - Computing flame: {label}", flush=True)

        # Set up conditions
        if composition_type == "zmixs":
            gas.TPY = oxid_temp, cond[0] * ct.one_atm, oxid_comp
            oxstream = ct.Quantity(gas, constant="HP")
            oxstream.mass = 1.0 - cond[1]
            gas.TPY = fuel_temp, cond[0] * ct.one_atm, fuel_comp
            fustream = ct.Quantity(gas, constant="HP")
            fustream.mass = cond[1]
            mix = fustream + oxstream
            gas.TPY = mix.T, mix.P, mix.Y
        elif composition_type == "phis":
            gas.TP = cond[1], cond[0] * ct.one_atm
            gas.set_equivalence_ratio(cond[2], fuel_comp, oxid_comp, basis="mass")
        else:
            gas.TPY = cond[1], cond[0] * ct.one_atm, comp

        # Solve Flame
        flame = ct.FreeFlame(gas, width=flame_width)
        flame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
        flame.set_max_grid_points(1, max_points)
        flame.transport_model = transport
        flame.solve(loglevel=loglevel)

        # Extract outputs from flame
        max_dtempdx = np.max(np.diff(flame.T) / np.diff(flame.grid))
        flame_thickness = (np.max(flame.T) - np.min(flame.T)) / max_dtempdx
        flame_temp = flame.T[-1]
        flame_speed = flame.velocity[0]
        flame_grid = len(flame.T)
        dx_min = np.min(np.diff(flame.grid))
        output.loc[cond] = flame_speed, flame_temp, flame_thickness, flame_grid, dx_min

        # Save flame solution - default Cantera MKS units
        data = pd.DataFrame()
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
        spec_names = [f"Y-{spec}" for spec in gas.species_names]
        spec_y_data = pd.DataFrame(flame.Y.T, columns=spec_names)
        rr_names = [f"SRC_{spec}" for spec in gas.species_names]
        spec_rr_data = pd.DataFrame(
            flame.net_production_rates.T * list(gas.molecular_weights), columns=rr_names
        )
        data = pd.concat([data, spec_y_data, spec_rr_data], axis=1)
        data.to_csv(os.path.join(outdir, f"flame_{label}.csv"))

        # We're finished with this flame
        print(
            f"Rank {rank} - Finished  flame: {label}. "
            f"sL={flame_speed:7.4f} Tad={flame_temp:7.1f} l_f={flame_thickness:10.3e} "
            f"N={flame_grid:5n} dx_min={dx_min:10.3e}",
            flush=True,
        )

    # Collect data and save
    print(f"Rank {rank} - Completed all required tasks")
    all_output = comm.gather(output)
    if rank == 0:
        all_output = pd.concat(all_output).sort_index()
        all_output.to_csv(os.path.join(outdir, "flamestats.csv"))
        print("All Ranks Completed.")
        print(all_output)
        if metadata_file != "":
            with open(metadata_file, "w") as fi:
                fi.write("manifold.has_species_mw = true\n")
                for i, spec in enumerate(gas.species_names):
                    fi.write(f"manifold.{spec}_mw = {gas.molecular_weights[i]}\n")
                if len(pressures) != 1:
                    raise RuntimeError("Can only save metadata for a single pressure")
                fi.write("manifold.nominal_pressure_cgs = " + str(pressures[0] * 10.0))
