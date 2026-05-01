"""
Script to compile 1D nonpremixed flames into a chemtable.

Uses output from the compute_nonpremixed_flamelets.py script. This script can generate
tables 1, 2, 3, or 4 dimensional tables, which miust always include mixture fraction
'ZMIX' as one of the table dimensions. Other dimensions may include progress variable
'PROG', mixture fraction variance 'ZVAR', and one additional mixture parameter for
multi-stream systems ('Z2MIX', 'FMIX' (two streams mix to form fuel stream) or 'WMIX'
(two streams mix to form oxidixer stream)).
"""

if __name__ == "__main__":

    import numpy as np
    import glob

    from cmlm.utils import TomlParmParse
    from cmlm.utils.input_file import scalar_to_list

    # ------------------ Parse relevant inputs ----------------------------#
    pp = TomlParmParse.parse_args(
        description="A tool to compile a nonpremixed table (up to 4 dimensions)",
        require_output=True,
    )
    outdir = pp.output_dir

    ppo = pp.get("options", doc="General options for running script")
    use_mpi = ppo.get(
        "use_mpi",
        default=False,
        doc="Using MPI requires mpi4py. Only the initial interpolation/convolution"
        "step is parallelized.",
    )
    verbose = ppo.get("verbose", default=0, doc="verbosity level of terminal output")
    flamelet_files = ppo.get(
        "flamelet_files",
        doc="path to flamelet solutions (will glob all files matching this pattern)",
    )
    output_file = ppo.get(
        "output_file", default="nonpremixed.ctb", doc="name of output table file"
    )

    ppm = pp.get("model", doc="Set up for tabulated chemistry model")
    model_name = ppm.get(
        "model_name",
        default="Nonpremixed",
        doc="Model name to be included in table file",
    )
    table_var_options = ["ZMIX", "ZVAR", "PROG", "Z2MIX", "WMIX", "FMIX"]
    table_vars = scalar_to_list(
        ppm.get(
            "table_vars",
            doc=f"List of table dimensions to include, must be in {table_var_options}",
        )
    )
    # Chack all requested dimensions are allowable
    for var in table_vars:
        if var not in table_var_options:
            raise ValueError(
                f"Requested table dimension {var} not in allowble list: {table_var_options}"
            )
    # Ensure we have mixture fraction
    if "ZMIX" not in table_vars:
        raise ValueError(
            "Requested table dimensions must include 'ZMIX' for nonpremixed table"
        )
    # Ensure no duplicates in table dimensions
    table_dim = len(table_vars)
    if len(set(table_vars)) != table_dim:
        raise ValueError("List of requested table vars contains duplicates")
    # Ensure we have only one second mixing parameter
    if ("Z2MIX" in table_vars) + ("WMIX" in table_vars) + ("FMIX" in table_vars) > 1:
        raise ValueError(
            "Table dimensions can only include one of 'WMIX', 'FMIX', 'Z2MIX'"
        )

    # Get grid for each variable
    grids = {}
    for var in table_vars:
        grid_type = ppm.get(
            f"{var}_grid.type",
            default="from_data",
            choices=["linear", "array", "from_data"],
            doc=f"Method of defining {var} grid",
        )
        # Linear: specify array of bounds and array of grid sizes,
        # linspace used between consecutive values in bounds
        if grid_type == "linear":
            bounds = scalar_to_list(
                ppm.get(
                    f"{var}_grid.bounds",
                    doc=f"Bounds for {var} grid, may specify multiple ranges, "
                    "must be strictly increasing",
                )
            )
            if len(bounds) < 2:
                raise ValueError(f"Must specify at least 2 bounds for {var} grid")
            if not all(np.array(bounds[1:]) > np.array(bounds[:-1])):
                raise ValueError(f"Bounds for {var} grid must be strictly increasing")
            ngrid = scalar_to_list(
                ppm.get(
                    f"{var}_grid.ngrid",
                    doc="Number of grid points for linear segment (must specify "
                    "nbounds-1 integers)",
                )
            )
            if not len(ngrid) == len(bounds) - 1:
                raise ValueError(f"Must specify nbounds-1 grid sizes for {var} grid")
            for grid in ngrid:
                if grid <= 0:
                    raise ValueError(f"{var} grid: All grid sizes must be positive.")
            grid_segments = [[bounds[0]]] + [
                np.linspace(bounds[i], bounds[i + 1], ng)[1:]
                for i, ng in enumerate(ngrid)
            ]
            grids[var] = np.concatenate(grid_segments)
        # Array: any strictly increasing array of size two or greater is valid
        elif grid_type == "array":
            grids[var] = scalar_to_list(
                ppm.get(
                    f"{var}_grid.grid",
                    doc="Array of values defining grid for variable, must be "
                    "strictly increasing",
                )
            )
            if len(grids[var]) < 2:
                raise ValueError(
                    f"Must specify at least two grid values for {var} grid"
                )
            if not all(np.array(grids[var][1:]) > np.array(grids[var][:-1])):
                raise ValueError(
                    f"Grid values for {var} grid must be strictly increasing"
                )
        # We can infer the desired grid from data if a dimension is FMIX or WMIX
        else:
            allowable_from_data = ["FMIX", "WMIX"]
            if var not in allowable_from_data:
                raise ValueError(
                    f"Grids can only be inferred from data for table dimensions in"
                    f" {allowable_from_data}. For {var} specify an alternative grid "
                    f"definition."
                )
            grids[var] = None

    prog_def = ppm.get(
        "prog_definition",
        doc="Progress variable definition, mapping species mass fractions to their weights",
    ).to_dict()
    keep_vars = ppm.get("keep_vars", doc="List of variables to include in output table")

    # ---------------------------------------------------------#
    #                    Start of Main                         #
    # ---------------------------------------------------------#

    # Convolute/interpolate all flamelet files
    files = sorted(glob.glob(flamelet_files))[::-1] # must be in increasing Lambda (generalized progvar) order
    print(files)
