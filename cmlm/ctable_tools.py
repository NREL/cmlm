"""Tools for tabulated function data of arbitrary dimension (via pandas DataFrames)."""

import os
import struct

import numpy as np
import pandas as pd
from scipy.interpolate import interpn


def check_tabfunc_integrity(ctable, allow_monoindex=False):
    """
    Verify DataFrame can be a valid TabulatedFunction for interpolation.

    Parameters
    ----------
    ctable: DataFrame
        Table to check integrity of
    allow_monoindex: Bool
        If true, don't require DataFrame to MultiIndex

    Returns
    -------
    is_valid: bool
        True indicates table is ready for interpolation
    """
    # Must have a multi-index and float64 data - if not we can't go on
    if not allow_monoindex and type(ctable.index) != pd.core.indexes.multi.MultiIndex:
        raise RuntimeError(
            "TabulatedFunction index must be a pandas MultiIndex DataFrame"
        )
    try:
        ctable.astype(np.float64)
    except ValueError:
        raise ValueError("TabulatedFunction data must be floats")

    # All axes must be regular grids - if not we don't need to die, but can't interpolate
    test_index = pd.MultiIndex.from_product(ctable.index.levels)
    if len(test_index) != len(ctable.index):
        return False
    elif all(test_index == ctable.index):
        return True
    else:
        return False


def is_strictly_increasing(array):
    """Return true if array is strictly monotonically increasing."""
    return all(xnp1 > xn for xn, xnp1 in zip(array[:-1], array[1:]))


def interpolate_axis(
    ctable,
    axis_name,
    new_grid,
    grid_var_name=None,
    remove_nonmonotonic=False,
    verbose=0,
):
    """
    Interpolates one axis of a DataFrame/TabulatedFunction onto a new grid.

    We require that all axes to the right of the specified axis already be
    on regular cartesian grids (they will be interpolated simultaneously). The
    interpolation axis itself and all axes to the left may be on irregular
    grids. After interpolation, the interpolation axis will be regular, but the
    axes to the left won't be changed.


    Parameters
    ----------
    ctable: pandas DataFrame, TabulatedFunction
        Table that we will interpolate onto new_grid
    axis_name: str
        Name of table axis that will be interpolated
    new_grid: 1d array of floats (or similar object) sorted in increasing order
        Modified grid values on which to interpolate (must be strictly ascending)
    verbose: int
        Verbosity level of logging to terminal (0: none, 1: one statement,
        2: statement for each group)
    grid_var_name: str, default None
        If set, use the specified column of the table to interpolate/redefine the
        specified axis
    remove_nonmonotonic: Bool, (default False)
        If True, ignore nonmonotonic entries in the current grid; otherwise an error
        will be raised for non-monotonic values in the current grid

    Returns
    -------
    newtable: pandas DataFrame
        Table with specified axis_name interpolated onto the new_grid
    """
    if verbose > 0:
        addstr = ""
        if grid_var_name is not None:
            addstr = f" -> {grid_var_name}"
        print(f"   Interpolating axis: {axis_name}{addstr}")

    axis_ind = ctable.index.names.index(axis_name)

    # Error test new grid - must be monotonic ascending
    if not is_strictly_increasing(new_grid):
        raise RuntimeError("Requested grid must be monotonicly increasing and is not.")

    # will do the interpolation separately for each point in the axes left of the
    # interpolation axis (but grouped together for axes to the right)
    groups = (
        (((), ctable),)
        if (axis_ind == 0)
        else ctable.groupby(ctable.index.names[:axis_ind])
    )

    # Generate new axis indices and an empty dataframe for the output
    new_index_right = pd.MultiIndex.from_product(
        [new_grid] + [lev for lev in ctable.index.levels[axis_ind + 1 :]],
    )
    if axis_ind > 1:
        new_index = pd.MultiIndex.from_tuples(
            [key1 + key2 for key1 in groups.indices.keys() for key2 in new_index_right],
            names=ctable.index.names,
        )
    elif axis_ind == 1:
        new_index = pd.MultiIndex.from_tuples(
            [
                (key1,) + key2
                for key1 in groups.indices.keys()
                for key2 in new_index_right
            ],
            names=ctable.index.names,
        )
    else:
        new_index = new_index_right
        new_index.names = ctable.index.names
    if grid_var_name is not None:
        new_index.rename({axis_name: grid_var_name}, inplace=True)
    new_table = pd.DataFrame(
        data=0.0, index=new_index, columns=ctable.columns, dtype=np.float64
    )

    # Interpolate group by group
    for idx, subdata in groups:
        if verbose > 0:
            print(f"    |-> for: {new_table.index.names[:axis_ind]} = {idx}")
        new_subdata = new_table.loc[idx]

        # account for variability in axis_ind axis
        old_shape = list(subdata.index.levshape[axis_ind:] + (ctable.shape[1],))

        # Ensure that the new data will be on a regular grid
        check_tabfunc_integrity(new_subdata, allow_monoindex=True)

        # Current grid - Remove non-monotonic if requested. Otherwise interpn will error.
        if grid_var_name is None:
            current_grid = subdata.index.remove_unused_levels().levels[axis_ind]
        else:
            # new variable must have same value for each value of existing variable
            # across all indices to the right of the interpolation index
            current_grid = []
            for _, grp2 in subdata[grid_var_name].groupby(axis_name):
                val = grp2.iloc[0]
                current_grid.append(val)
                if not all(grp2 == val):
                    raise RuntimeError(
                        "If using grid_var_name, that variable must have same value"
                        "for all right indices"
                    )

        if not is_strictly_increasing(current_grid):
            if remove_nonmonotonic:
                mod_current_grid = []
                mod_grid_idx = []
                maxval = -np.inf
                ndrop = 0
                for gridval, idxval in zip(
                    current_grid, subdata.index.remove_unused_levels().levels[axis_ind]
                ):
                    if gridval > maxval:
                        maxval = gridval
                        mod_current_grid.append(gridval)
                        mod_grid_idx.append(idxval)
                    else:
                        ndrop += 1
                if verbose > 2:
                    print(
                        f"dropping {ndrop}/{len(current_grid)} points for nonmonotonicity"
                    )
                current_grid = mod_current_grid
                interpdata = subdata.droplevel(subdata.index.names[:axis_ind]).loc[
                    mod_current_grid
                ]
                old_shape[0] -= ndrop
            else:
                raise RuntimeError("Cannot interpolate because data are nonmonotone.")
        else:
            interpdata = subdata

        # clamp new grid to valid domain so we don't extrapolate OOB
        interp_grid = np.clip(new_grid, current_grid[0], current_grid[-1])

        # Do interpolation
        new_subdata[:] = interpn(
            (current_grid,),  # existing grid for interpolation axis
            interpdata.to_numpy().reshape(
                old_shape
            ),  # existing data, reshaped to match the axis shape
            interp_grid,  # new grid for interpolation axis
            bounds_error=False,  # allow out of bounds data
            fill_value=None,  # Use FOextrap to fill out of bounds data
        ).reshape(new_subdata.shape)

    return new_table


def convert_chemtable_units(ctable, conversion="mks2cgs"):
    """
    Find selected variables in a table and convert units between CGS and MKS.

    Converts variables with the following names (mks2cgs conversion shown):

    - `RHO`: kg m-3 -> g cm-3

    - `DIFF` (actually rhoD): kg s-1 m-1 -> g s-1 cm-1

    - `VISC` (dynamic): kg s-1 m-1 -> g s-1 cm-1

    - `WBAR` (molecular mass): kg mol-1 -> g mol-1

    - `SRC_*` (species source terms): kg m-3 s-1 -> g cm-3 s-1

    - `T`: K -> K

    - `X` (length): m -> cm

    - `VEL` (velocity) m s-1 -> cm s-1

    Parameters
    ----------
        ctable: TabulatedFunction, pd.DataFrame
            tabular data to convert (conversion happens in place)
        conversion: str ('mks2cgs' or 'cgs2mks', default 'mks2cgs')
            unit conversion to perform

    Returns
    -------
        ierr: int
            conversion happens in place, returns 0 if success
    """
    # MKS to CGS conversion factors
    conversions = {
        "RHO": 1.0e-3,  # kg m-3 -> g cm-3
        "DIFF": 10.0,  # (rhoD) kg s-1 m-1 -> g s-1 cm-1
        "VISC": 10.0,  # (dynamic) kg s-1 m-1 -> g s-1 cm-1
        "WBAR": 0.001,  # (molecular mass) kg/mol -> g/mol
        "SRC_": 1.0e-3,  # source terms kg m-3 s-1 -> g cm-3 s-1
        "T": 1.0,  # K -> K
        "X": 1.0e2,  # m -> cm
        "VEL": 1.0e2,  # m s-1 -> cm s-1
    }

    if conversion not in ["mks2cgs", "cgs2mks"]:
        raise RuntimeError(
            "convert_chemtable_units: can only convert mks2cgs or cgs2mks"
        )

    if conversion == "cgs2mks":
        for var in conversions.keys():
            conversions[var] = 1 / conversions[var]

    for var in ctable.columns:
        varmod = var if not var.startswith("SRC_") else "SRC_"
        if varmod in conversions.keys():
            ctable[var] *= conversions[varmod]

        elif not var.startswith("Y-"):
            # Warn if not a mass fraction and no conversion is found
            print(f"WARNING: no conversion for tabulated variable {var}")

    return 0


def print_chemtable(df, model_name=None):
    try:
        mod_name = df.model_name
    except AttributeError:
        mod_name = model_name
    print(TabulatedFunction(df, model_name=mod_name))


def read_chemtable_binary(
    filename, tformat="Pele", Ndim=None, Dimnames=None, verbose=0
):

    # check inputs
    if tformat not in ["Pele", "NGA"]:
        raise RuntimeError("Invalid table format")
    if tformat == "NGA":
        if Ndim is None:
            raise RuntimeError("Must specify number of dimensions for NGA format")

    with open(filename, "rb") as fi:
        if tformat == "Pele":
            Ndim = struct.unpack("i", fi.read(4))[0]
            dim_names = [
                struct.unpack("64s", fi.read(64))[0].decode().strip()
                for idim in range(Ndim)
            ]
        else:
            if Dimnames is None:
                dim_names = ["dim" + str(idim) for idim in range(Ndim)]
            else:
                if len(Dimnames) != Ndim:
                    raise RuntimeError("Wrong number of dim names given")
                dim_names = Dimnames

        dimLengths = struct.unpack(str(Ndim) + "i", fi.read(Ndim * 4))
        Nvar = struct.unpack("i", fi.read(4))[0]
        grids = [
            struct.unpack(str(dimLengths[idim]) + "d", fi.read(dimLengths[idim] * 8))
            for idim in range(Ndim)
        ]
        model_name = struct.unpack("64s", fi.read(64))[0].decode().strip()
        var_names = [
            str(struct.unpack("64s", fi.read(64))[0].decode().strip())
            for ivar in range(Nvar)
        ]
        Ndata_var = np.prod(list(dimLengths))
        Ndata = Nvar * Ndata_var
        data = np.array(struct.unpack(str(Ndata) + "d", fi.read(Ndata * 8))).reshape(
            Ndata_var, Nvar, order="F"
        )

    ctable_index = pd.MultiIndex.from_product(
        reversed(grids), names=reversed(dim_names)
    )
    ctable = pd.DataFrame(data, index=ctable_index, columns=var_names)

    return ctable, model_name


def write_chemtable_binary(filename, ctable, tablename, tformat="Pele"):

    with open(filename, "wb") as fi:

        Ndim = len(ctable.index.names)
        if tformat == "Pele":
            # Number of Dimensions
            fi.write(struct.pack("i", Ndim))
            # Dimension Names
            fi.write(
                struct.pack(
                    str(Ndim * 64) + "s",
                    "".join(
                        [f"{name:<64s}" for name in reversed(ctable.index.names)]
                    ).encode(),
                )
            )

        # Dimension Lengths
        fi.write(
            struct.pack(
                str(Ndim) + "i",
                *[len(level) for level in reversed(ctable.index.levels)],
            )
        )

        # Number of Variables
        fi.write(struct.pack("i", len(ctable.columns)))

        # Grids
        for ii in reversed(range(Ndim)):
            fi.write(np.array(ctable.index.levels[ii]).tobytes())

        # Model Name
        fi.write(struct.pack("64s", tablename.encode()))

        # Variable Names
        fi.write(
            struct.pack(
                str(len(ctable.columns) * 64) + "s",
                "".join([f"{name:<64s}" for name in ctable.columns]).encode(),
            )
        )

        # Data
        for col in ctable.columns:
            assert (
                ctable.dtypes[col] == np.float64
            ), "Chemtable data type must be float64"
            fi.write(ctable[col].to_numpy().tobytes())


def slice_table(ctable, slice_vars=None, slice_vals=None, slice_pairs=None):
    # create a slice of a chemtable
    # must specify lists of slice_vars and slice_vals or slice_pairs of format
    # ["var0:val0", "var1:val1", "var2:val2"]
    # slices in dimensions specified by slice_vars at locations specified by slice_vals
    if slice_pairs is not None:
        assert slice_vars is None and slice_vals is None
        slice_vars = [pair.split(":")[0] for pair in slice_pairs]
        slice_vals = [float(pair.split(":")[1]) for pair in slice_pairs]
    else:
        assert slice_vars is not None and slice_vals is not None
    assert len(slice_vars) == len(slice_vals)
    slice_val_dict = dict(zip(slice_vars, slice_vals))
    found_vars = []
    found_slice_vals = []
    for var in slice_vars:
        assert var in ctable.index.names
        assert var not in found_vars  # no repeats allowed
        found_vars.append(var)
        vals = ctable.index.get_level_values(var)
        closest_val = vals[np.argmin(np.abs(vals - slice_val_dict[var]))]
        print(
            "Slice variable "
            + var
            + " at "
            + str(closest_val)
            + " (requested "
            + str(slice_val_dict[var])
            + ")"
        )
        found_slice_vals.append(closest_val)
    sliced_table = ctable[
        ctable.index.get_loc_level(found_slice_vals, slice_vars)[0]
    ].reset_index(level=slice_vars, drop=True)
    if sliced_table.index.nlevels == 1:
        sliced_table.index = pd.MultiIndex.from_arrays([sliced_table.index])
    return sliced_table


class TabulatedFunction(pd.DataFrame):

    def __init__(
        self,
        table,
        model_name=None,
        verbose=0,  # General inputs
        tformat="Pele",
        Ndim=None,
        Dimnames=None,  # if reading from file
        index=None,
        columns=None,
        dtype=None,
        copy=None,  # pass to inherited dataframe constructor
    ):

        if isinstance(table, str):
            # if input is a string, treat as a path to a file to read
            if verbose > 0:
                print("Reading Table from file: ", table)
            mydf, myname = read_chemtable_binary(
                table, tformat, Ndim, Dimnames, verbose
            )
            super().__init__(mydf)
            self.model_name = myname

        else:
            # otherwise, use the parent (DataFrame) constructor
            if verbose > 0:
                print("Creating chemtable from Pandas MultiIndex dataframe")
            super().__init__(
                table, index=index, columns=columns, dtype=dtype, copy=copy
            )
            self.model_name = model_name

        # save verbosity and print some stuff if requested
        self.verbose = verbose
        if self.verbose > 0:
            print("Successfully created table")
            print(self)

        self.saved_is_valid = None
        self.is_valid()

    def __str__(self):
        """Write table summary to a string."""
        out = "\n"
        out += "--- TABULATED FUNCTION ---" + "\n"
        out += "\n"
        out += f"Model Name: {self.model_name} \n"
        out += "\n"
        out += f"Dimensions ({self.getNdim()}):\n"
        for ii, dim in enumerate(self.getDimNames()):
            out += f"    Dim: {ii:<2d} Name: {dim:<10s} Length: {len(self.index.levels[ii])}\n"
            out += "         Values:" + "\n"
            out += (
                " "
                + " ".join(
                    [
                        ("         " if ii % 4 == 0 else "")
                        + f"{val:16.8e}"
                        + ("\n" if ii % 4 == 3 else "")
                        for ii, val in enumerate(self.index.levels[ii])
                    ]
                )
                + "\n"
            )
        out += "\n"
        out += f"Variables ({len(self.columns)}):\n"
        for ii, var in enumerate(self.columns):
            out += (
                f"    Var: {ii:<2d} Name: {var:<10s} Min: {np.min(self[var]):16.8e}"
                f" Max: {np.max(self[var]):16.8e}\n"
            )
        out += "\n"
        if self.verbose > 0:
            out += super().__str__()
        return out

    def is_valid(self):
        if self.saved_is_valid is None:
            self.saved_is_valid = check_tabfunc_integrity(self)
            return self.saved_is_valid
        else:
            return self.saved_is_valid

    def getNdim(self):
        return len(self.index.names)

    def getDimNames(self):
        return self.index.names

    def getDimSizes(self):
        return [len(grid) for grid in self.index.levels]

    def getMatrixData(self, var):
        return self[var].to_numpy().reshape(self.getDimSizes())

    def interpolate(self, var, vals=None, method="linear", **kwargs):

        if not self.is_valid():
            raise RuntimeError("Trying to interpolate with an invalid table")

        # vals is an array of variables in order
        # each variable may be a scalar or an array of values
        if vals is not None:
            lookup = np.array(vals)

        else:  # take vals from kwargs
            lookup = []
            for name in self.getDimNames():
                if name not in kwargs.keys():
                    raise RuntimeError(
                        f"table dim <{name}> not specified in interpolate functions"
                    )
                lookup.append(kwargs[name])
            lookup = np.array(lookup)

        if lookup.shape[0] != self.getNdim():
            raise RuntimeError(
                "TabulatedFunction.interpolate(): number of vals passed must equal "
                "number of table dimensions"
            )

        out = interpn(
            self.index.levels,
            self.getMatrixData(var),
            lookup.T,
            method=method,
            bounds_error=False,
            fill_value=None,
        )
        return out.T


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Useful tools for interacting with chemtable files"
    )
    parser.add_argument("inputfile", help="Table file to read", type=str)
    parser.add_argument(
        "-f",
        "--format",
        dest="tformat",
        choices=["NGA", "Pele"],
        default="Pele",
        help="Table format, NGA or Pele",
    )
    parser.add_argument(
        "-d", "--dimensions", type=int, help="Number of dimensions in NGA table"
    )
    parser.add_argument(
        "-dl",
        "--dimension_labels",
        type=str,
        nargs="+",
        help="List of dimension names in NGA table",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print_table",
        action="store_true",
        help="Flag to print extrema of table file",
    )
    parser.add_argument(
        "-o", "--outputfile", type=str, help="Output file to write, if needed"
    )
    parser.add_argument(
        "-cf",
        "--convert_format",
        action="store_true",
        help="Convert format from input format to other format",
    )
    parser.add_argument(
        "-cu",
        "--convert_units",
        type=str,
        choices=["none", "cgs2mks", "mks2cgs"],
        default="none",
        help="Convert units and save new table",
    )
    parser.add_argument(
        "-sl",
        "--slice",
        type=str,
        nargs="+",
        help="Create a table by slicing. Specify dimensions to slice "
        "and values as a list of form "
        "dim_name1:value dim_name2:value",
    )
    parser.add_argument(
        "-sp",
        "--slice_plot",
        type=str,
        nargs="+",
        help="Plot a slice of the table. Specify dimensions to slice"
        " and values as a list of form dim_name1:value dim_name2:value,"
        " must specify enough slice dims such that there are exactly"
        " one or two slice dimensions. Alternatively, set as 'justplot'"
        " to directly plot a 1D or 2D table without slicing.",
    )
    parser.add_argument(
        "-v",
        "--variables",
        type=str,
        nargs="+",
        help="variables to be plotted"
        "if unspecified, all variables in table will be plotted",
    )
    args = parser.parse_args()

    ctable, tname = read_chemtable_binary(
        args.inputfile, args.tformat, args.dimensions, args.dimension_labels
    )

    if args.print_table:
        print_chemtable(ctable)
        print(ctable)

    convert = (
        args.convert_format or (args.convert_units != "none") or args.slice is not None
    )
    if convert:
        if args.outputfile is None:
            raise RuntimeError(
                "Output file must be specified for table format/units/slicing conversion"
            )

        if args.slice is not None:
            out_table = slice_table(ctable, slice_pairs=args.slice)
        else:
            out_table = ctable

        if args.convert_units != "none":
            convert_chemtable_units(out_table, conversion=args.convert_units)

        if args.convert_format:
            output_format = "Pele" if args.tformat == "NGA" else "NGA"
        else:
            output_format = args.tformat

        write_chemtable_binary(args.outputfile, out_table, tname, output_format)

    if args.slice_plot is not None:
        if args.slice_plot[0] != "justplot":
            plot_dims = ctable.index.nlevels - len(args.slice_plot)
            if not (os.path.exists(args.outputfile)):
                os.makedirs(args.outputfile)
            sliced = slice_table(ctable, slice_pairs=args.slice_plot)
        else:
            plot_dims = ctable.index.nlevels
            sliced = ctable

        assert plot_dims > 0 and plot_dims <= 2
        assert args.outputfile is not None
        if not os.path.exists(args.outputfile):
            os.makedirs(args.outputfile)
        plt_vars = args.variables if args.variables is not None else ctable.columns

        import matplotlib.pyplot as plt

        if plot_dims == 1:
            for var in plt_vars:
                plt.figure()
                plt.plot(sliced.index.levels[0], sliced[var], "r-")
                plt.xlabel(sliced.index.names[0])
                plt.ylabel(var)
                plt.savefig(
                    os.path.join(
                        args.outputfile,
                        "1Dslice_"
                        + "_".join([pair.replace(":", "") for pair in args.slice_plot])
                        + "_"
                        + var
                        + "_vs_"
                        + sliced.index.names[0]
                        + ".png",
                    )
                )
                plt.clf()
                plt.close()
        elif plot_dims == 2:
            for var in plt_vars:
                plt.figure()
                plt.contourf(
                    sliced.index.levels[1],
                    sliced.index.levels[0],
                    sliced[var].to_numpy().reshape(sliced.index.levshape),
                    levels=100,
                )
                plt.xlabel(sliced.index.names[1])
                plt.ylabel(sliced.index.names[0])
                plt.colorbar(label=var)
                plt.savefig(
                    os.path.join(
                        args.outputfile,
                        "2Dslice_"
                        + "_".join([pair.replace(":", "") for pair in args.slice_plot])
                        + "_"
                        + var
                        + "_vs_"
                        + sliced.index.names[0]
                        + "_"
                        + sliced.index.names[1]
                        + ".png",
                    ),
                    dpi=150,
                )
                plt.clf()
                plt.close()


if __name__ == "__main__":

    main()
