"""A module for interacting with (loading data from) input files."""

import sys

import toml


class TomlParmParse:
    """
    Query or get parameters from a TOML input file.

    Inspired by the ParmParse class from AMReX

    Parameters
    ----------
        file_name: stirng
            path to input file
        allow_cl_override: bool, optional
            if True, sys.argv[1] replaces `file_name`. Default True.
    """

    def __init__(self, file_name=None, allow_cl_override=True):

        if file_name is None and (len(sys.argv) <= 1 or not allow_cl_override):
            raise RuntimeError(
                "TomlParmParse: must provide input file to initializer or on command_line"
            )

        if len(sys.argv) > 1 and allow_cl_override:
            load_file = sys.argv[1]
        else:
            load_file = file_name

        try:
            with open(load_file) as tomlfile:
                self.data = toml.load(tomlfile)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"TomlParmParse: Input file < {load_file} > not found!"
            )

    def query(self, prefix, var, default):
        """
        Look up a value from the input file, if not present use default.

        Parameters
        ----------
           prefix: string
              section of TOML file
           var: string
              entry in TOML file
           default: any type
              default to use if entry not found

        Returns
        -------
           value: any type
              value from TOML file or default if not present
        """
        if var in self.data[prefix].keys():
            return self.data[prefix][var]
        else:
            return default

    def get(self, prefix, var):
        """
        Look up a value from the input file, if not present raise error.

        Parameters
        ----------
           prefix: string
              section of TOML file
           var: string
              entry in TOML file

        Returns
        -------
           value: any type
              value from TOML file
        """
        if var in self.data[prefix].keys():
            return self.data[prefix][var]
        else:
            raise RuntimeError(
                f"TomlParmParse: Requested Variable: {prefix}.{var} not found in input file"
            )
