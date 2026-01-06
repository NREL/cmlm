"""
Script to solve 1D nonpremixed flames at varied conditions using Cantera.

The user specified physical conditions including pressure (constant) and
the temperature and composition of up to three streams: fuel, oxidizer,
and a dilution stream that is mixed with one of these streams prior
to reaction. Supports computations with varied mechanisms, transport
models, and equations of state (all handled through Cantera).

Uses TOML format input files (see example input file for more details)
and requires specification of an output directory on the command line::

   python compute_nonpremixed_flamelets.py compute_nonpremixed_flamelets.toml -o output
"""

if __name__ == "__main__":

    import itertools
    import os

    import numpy as np
    import pandas as pd
    from cmlm.utils import TomlParmParse
    from cmlm.utils.cantera_helpers import save_flame_csv, save_table_metadata
    from cmlm.utils.input_file import scalar_to_list
    from func_timeout import FunctionTimedOut, func_timeout

    import cantera as ct

    def update_flame(flame, strain_factor):
        """Create inital guess for flame after chainging strain rate."""
        # Exponents for the initial solution variation with changes in strain rate
        # Taken from Fiala and Sattelmayer (2014)
        exp_d_a = -1.0 / 2.0
        exp_u_a = 1.0 / 2.0
        exp_V_a = 1.0
        exp_lam_a = 2.0
        exp_mdot_a = 1.0 / 2.0
        flame.flame.grid *= strain_factor**exp_d_a
        flame.fuel_inlet.mdot *= strain_factor**exp_mdot_a
        flame.oxidizer_inlet.mdot *= strain_factor**exp_mdot_a
        flame.flame.set_values("velocity", flame.velocity * strain_factor**exp_u_a)
        flame.flame.set_values("spreadRate", flame.spread_rate * strain_factor**exp_V_a)
        flame.flame.set_values("lambda", flame.L * strain_factor**exp_lam_a)

    # ------------------ Parse relevant inputs ----------------------------#
    pp = TomlParmParse.parse_args(
        description="A tool to compute nonpremixed counterflow flames using Cantera",
        require_output=True,
    )
    outdir = pp.output_dir

    # Conditions
    ppc = pp["conditions"].doc("Physical conditions for flames")
    pressures = scalar_to_list(ppc.get("pressure", doc="atm"))
    fuel_temp = ppc.get("fuel_temp", doc="K")
    fuel_comp = ppc.get("fuel_comp", doc="Fuel stream cantera composition, mass basis")
    oxid_temp = ppc.get("oxid_temp", doc="K")
    oxid_comp = ppc.get(
        "oxid_comp", doc="Oxidizer stream cantera composition, mass basis"
    )
    stream_to_dilute = ppc.get(
        "stream_to_dilute",
        default="none",
        choices=["none", "fuel", "oxid"],
        doc="Add dilution to fuel stream, oxid stream, or none",
    )
    if stream_to_dilute != "none":
        dilu_temp = ppc.get("dilu_temp", doc="K")
        dilu_comp = ppc.get(
            "dilu_comp", doc="Dilution stream cantera composition, mass basis"
        )
        dilu_bounds = ppc.get(
            "dilu_fraction_bounds",
            default=[0, 1],
            doc="Array with upper and lower dilution values",
        )
        dilu_n_grid = ppc.get(
            "dilu_n_grid", default=21, doc="Number of dilution values to use"
        )
        dilu_grid = np.linspace(dilu_bounds[0], dilu_bounds[1], dilu_n_grid)
    mdot_initial = ppc.get(
        "mdot_initial",
        default=0.1,
        doc="Initial mass flow rate for counterflow configuration",
    )
    flame_width = ppc.get(
        "dom_width_initial",
        default=1.0,
        doc="Initial domain width for counterflow configuration (m)",
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
    tols = ppf.get("tols", default=[1e-6, 1e-12])

    # S-curve Options
    pps = pp["scurve"].doc("Options for traversing S-curves")
    delta_temperature_limit_extinction = pps.get(
        "delta_T_extinction",
        default=50,
        doc="Minimum T rise (K) for flamelet to be considered burning",
    )
    delta_alpha_initial = pps.get(
        "delta_alpha_initial",
        default=1.0,
        doc="Initial change in strain rate factor between successive flamelets",
    )
    delta_alpha_min = pps.get("delta_alpha_min", default=0.0025, doc="")
    delta_alpha_max = pps.get("delta_alpha_max", default=3.0, doc="")
    delta_alpha_max_change_factor = pps.get(
        "delta_alpha_max_change_factor", default=3.0, doc=""
    )
    delta_temp_min = pps.get("delta_temp_min", default=1.0, doc="")
    delta_temp_max_initial = pps.get("delta_temp_max", default=20.0, doc="")
    max_solve_time = pps.get(
        "max_solve_time",
        default=60,
        doc="s, Flames will be treated as unconverged if solve not completed in this time",
    )

    # Other options
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

    # ------------------ Set up flames to run ---------------------------- #
    cond_labels = ["p{:.4f}"]  # noqa : FS003
    if stream_to_dilute == "fuel":
        cond_labels += ["F{:.4f}"] # noqa : FS003
        cond_iterator_global = list(itertools.product(pressures, dilu_grid))
    elif stream_to_dilute == "oxid":
        cond_labels += ["W{:.4f}"] # noqa : FS003
        cond_iterator_global = list(itertools.product(pressures, dilu_grid))
    else:
        cond_iterator_global = list(itertools.product(pressures))
    cond_iterator = cond_iterator_global[rank::nprocs]

    for cond in cond_iterator:
        label = "_".join(condlabel.format(c) for condlabel, c in zip(cond_labels, cond))
        print(f"Rank {rank}: Computing strain rate sweep for: {label}", flush=True)

        # Set up mixtures and flames
        press = cond[0] * ct.one_atm
        dilfact = cond[1]
        oxid = ct.Solution(mechanism, name=eos)
        oxid.TPY = oxid_temp, press, oxid_comp
        oxidstream = ct.Quantity(oxid, constant="HP")
        fuel = ct.Solution(mechanism, name=eos)
        fuel.TPY = fuel_temp, press, fuel_comp
        fuelstream = ct.Quantity(fuel, constant="HP")
        if stream_to_dilute != "none":
            dilu = ct.Solution(mechanism, name=eos)
            dilu.TPY = dilu_temp, press, dilu_comp
            dilustream = ct.Quantity(dilu, constant="HP")
            dilustream.mass = dilfact
        oxmix = oxidstream
        fumix = fuelstream
        if stream_to_dilute == "fuel":
            fuelstream.mass = 1.0 - dilfact
            fumix = fuelstream + dilustream
        elif stream_to_dilute == "oxid":
            oxidstream.mass = 1.0 - dilfact
            oxmix = oxidstream + dilustream

        # Create and Solve initial flame
        gas = ct.Solution(mechanism, eos)
        if metadata_file != "":
            gas.P = press
            save_table_metadata(gas, os.path.join(outdir, metadata_file))
        flame = ct.CounterflowDiffusionFlame(gas, width=flame_width)
        flame.P = press
        flame.fuel_inlet.Y = fumix.Y
        flame.fuel_inlet.T = fumix.T
        flame.fuel_inlet.mdot = mdot_initial
        flame.oxidizer_inlet.Y = oxmix.Y
        flame.oxidizer_inlet.T = oxmix.T
        flame.oxidizer_inlet.mdot = mdot_initial

        temperature_limit_extinction = (
            max(fumix.T, oxmix.T) + delta_temperature_limit_extinction
        )
        flame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)
        flame.flame.set_steady_tolerances(default=tols)
        flame.transport_model = transport
        flame.solve(loglevel=loglevel, auto=True)

        print(f"Rank {rank}: Creating the initial solution", flush=True)
        flame.solve(loglevel=loglevel, auto=True)

        n_init = 1000
        n = n_init
        n_last_burning = n
        file_name = os.path.join(outdir, f"nonp_{label}_{n:04d}")
        save_flame_csv(flame, f"{file_name}.csv", cp_fuel_species)
        flame.save(f"{file_name}.yaml", name=f"solution_{label}")

        # S-Curve Calculations
        # from: https://cantera.org/examples/python/onedim/diffusion_flame_extinction.py.html

        # Set normalized initial strain rate
        alpha = [np.nan] * n_init + [1.0]
        # Factor of refinement of the strain rate increase
        delta_alpha_factor = 4.0
        delta_alpha = delta_alpha_initial
        delta_temp_max = delta_temp_max_initial

        # List of peak temperatures
        T_max = [np.nan] * n_init + [np.max(flame.T)]
        # List of maximum axial velocity gradients
        a_max = [np.nan] * n_init + [
            np.max(np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid)))
        ]

        print(
            f"rank {rank}: Flame burning at alpha = {alpha[-1]:8.4F} "
            f"with Tmax = {T_max[-1]:06.1F}, Ngrid = {len(flame.grid):5d}. "
            f"Proceeding to the next iteration, delta_alpha = {delta_alpha}",
            flush=True,
        )

        # Simulate counterflow flames at increasing strain rates until the flame is
        # extinguished. To achieve a fast simulation, an initial coarse strain rate
        # increase is set. This increase is reduced after an extinction event and
        # the simulation is again started based on the last burning solution.
        # The extinction point is considered to be reached if the abortion criteria
        # on strain rate increase and peak temperature decrease are fulfilled.
        while True:
            # Update relative strain rates and initial guess
            n += 1
            alpha.append(alpha[n_last_burning] * (1 + delta_alpha))
            strain_factor = alpha[-1] / alpha[n_last_burning]
            update_flame(flame, strain_factor)

            # Solve Flame - if does not converge or times out, assume we are extinct
            solve_failed = False
            try:
                func_timeout(max_solve_time, flame.solve, kwargs={"loglevel": loglevel})
            except (ct.CanteraError, FunctionTimedOut) as e:
                solve_failed = True
                print(
                    f"rank {rank}: Did not converge at n = {n} with Error: {e}",
                    flush=True,
                )

            # Stop or continue based on criteria
            if np.max(flame.T) > temperature_limit_extinction and not solve_failed:
                # Flame is still burning, so proceed to next strain rate
                n_last_burning = n
                file_name = os.path.join(outdir, f"nonp_{label}_{n:04d}")
                flame.save(f"{file_name}.yaml", name=f"solution_{label}")
                save_flame_csv(flame, f"{file_name}.csv", cp_fuel_species)
                T_max.append(np.max(flame.T))
                a_max.append(
                    np.max(
                        np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid))
                    )
                )
                delta_alpha = max(
                    min(
                        delta_alpha
                        * max(
                            min(
                                delta_temp_max / (T_max[-2] - T_max[-1]),
                                delta_alpha_max_change_factor,
                            ),
                            1 / delta_alpha_max_change_factor,
                        ),
                        delta_alpha_max,
                    ),
                    delta_alpha_min,
                )
                print(
                    f"rank {rank}: Flame burning at alpha = {alpha[-1]:8.4F} "
                    f"with Tmax = {T_max[-1]:06.1F}, Ngrid = {len(flame.grid):5d}. "
                    f"Proceeding to the next iteration, delta_alpha = {delta_alpha}",
                    flush=True,
                )

            elif delta_alpha <= delta_alpha_min:
                # If the temperature difference is too small and the minimum relative
                # strain rate increase is reached, save the last, non-burning, solution
                # to the output file and break the loop
                T_max.append(np.max(flame.T))
                a_max.append(
                    np.max(
                        np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid))
                    )
                )
                file_name = os.path.join(outdir, f"nonp_{label}_final_{n:04d}")
                flame.save(f"{file_name}.yaml", name=f"solution_{label}")
                save_flame_csv(flame, f"{file_name}.csv")
                print(
                    f"rank {rank}: Flame extinguished at alpha = {alpha[-1]:8.4F}, "
                    "Stopping criteria satisfied.",
                    flush=True,
                )
                break

            else:
                # Procedure if flame extinguished but abortion criterion is not satisfied
                # Reduce relative strain rate increase
                delta_alpha = max(delta_alpha / delta_alpha_factor, delta_alpha_min)
                delta_temp_max = max(delta_temp_max / delta_alpha_factor, delta_temp_min)
                print(
                    f"rank {rank}: Flame extinguished at alpha = {alpha[-1]:8.4F}. "
                    f"Restoring alpha = {alpha[n_last_burning]:8.4F} and "
                    f"trying delta_alpha = {delta_alpha}",
                    flush=True,
                )
                # Restore last burning solution
                file_name = os.path.join(
                    outdir, f"nonp_{label}_{n_last_burning:04d}.yaml"
                )
                flame.restore(file_name, name=f"solution_{label}")

        # Save data for S curve
        pd.DataFrame({"a_max": a_max, "T_max": T_max}).dropna(how='all').to_csv(
            os.path.join(outdir, f"scurve_info_{label}.csv")
        )

        # Simulate counterflow flames at decreasing strain rates toward equilibrium
        # Reload initial flame to start
        file_name = os.path.join(outdir, f"nonp_{label}_{n_init:04d}.yaml")
        flame.restore(file_name, name=f"solution_{label}")
        n = n_init

        print(
            f"rank {rank}: Computing flamelets with lower strain toward equilibrium.",
            flush=True,
        )
        while True:
            n -= 1
            if n < 0:
                raise RuntimeError(
                    "Required too many low strain flames, ran out of room"
                )

            # Update relative strain rates and initial guess
            delta_alpha = 0.4
            strain_factor = 1 / (1 + delta_alpha)
            alpha[n] = alpha[n + 1] * strain_factor
            update_flame(flame, strain_factor)

            # Solve flame with new strain rate
            try:
                func_timeout(max_solve_time, flame.solve, kwargs={"loglevel": loglevel})
            except (ct.CanteraError, FunctionTimedOut) as e:
                print(
                    f"rank {rank}: Did not converge at n = {n} with Error: {e}",
                    flush=True,
                )
                break

            file_name = os.path.join(outdir, f"nonp_{label}_{n:04d}")
            flame.save(f"{file_name}.yaml", name=f"solution_{label}")
            save_flame_csv(flame, f"{file_name}.csv")

            T_max[n] = np.max(flame.T)
            a_max[n] = np.max(
                np.abs(np.gradient(flame.velocity) / np.gradient(flame.grid))
            )
            print(
                f"rank {rank}: Flame burning at alpha = {alpha[n]:8.4F} "
                f"with Tmax = {T_max[n]:06.1F}, Ngrid = {len(flame.grid):5d}. "
                f"Proceeding to the next iteration, delta_alpha = {delta_alpha}",
                flush=True,
            )
            if T_max[n] - T_max[n + 1] < delta_temp_min :
                print(f"rank {rank}: Reached Equilibrium - Stopping.")
                break

        # Save data for S curve
        pd.DataFrame({"a_max": a_max, "T_max": T_max}).dropna(how='all').to_csv(
            os.path.join(outdir, f"scurve_info_{label}.csv")
        )
