import sys

import toml


class TomlParmParse:

    def __init__(self, file_name=None, allow_cl_override=True):

        if file_name is None and (len(sys.argv) <= 1 or not allow_ck_override):
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
        if var in self.data[prefix].keys():
            return self.data[prefix][var]
        else:
            return default

    def get(self, prefix, var):
        if var in self.data[prefix].keys():
            return self.data[prefix][var]
        else:
            raise RuntimeError(
                f"TomlParmParse: Requested Variable: {prefix}.{var} not found in input file"
            )
