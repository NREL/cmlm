"""A module for interacting with TOML format input files based on tomlkit."""

import argparse
import os

import tomlkit


class TomlParmParse:
    """
    Wrapper for tomlkit document class.

    Enables a few new features:
    - Easier access to nested parameters
    - Autodocumentation of inputs
    - Combining inputs from the command line and an input file
    - Saving the config that was actually used
    - Optionally raise errors for unused inputs

    Inspired by the ParmParse class from AMReX, but quite different.

    Parameters
    ----------
        datadict: tomlkit.TOMLDocument
            input TOMLDocument that has been read in
        accessed_data: tomlkit.TOMLDocument, optional
            shows which entries from data_dict have already been accessed
            Default None.
        is_base: bool, optional
            If True, optional checks and file dumping occur during garbage collecting.
            Default False.
        output: string, optional
            Directory in which to save output. Default None (no output saved).
        output_type: str, optional
            Type of output to save: "clean" will save only inputs used with no comments.
            "doc" will save only inputs used with comments generated based on doc info
            provided when accessing variables. "original" keeps all variables, comments,
            and formatting from the provided input file. Default "clean".
        no_overwrite: bool, optional
            Raise an error if multiple different values are set/accessed for a variable.
            Default True.
        error_unused: bool, optional
            Raise an error for unused variables in input file. Default False.
    """

    def __init__(
        self, datadict, accessed_data=None, name="<base>", is_base=False, **kwargs
    ):
        self.data = datadict

        if accessed_data is not None:
            self.accessed_data = accessed_data
        else:
            accessed_data = tomlkit.document()
            self.accessed_data = accessed_data

        self.name = name
        self.is_base = is_base
        self.output = kwargs.get("output", None)
        self.output_dir = os.path.split(self.output)[0]
        if is_base:
            if len(self.output_dir) > 0 and not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        self.output_type = kwargs.get("output_type", "clean")
        self.no_overwrite = kwargs.get("no_overwrite", True)
        self.error_unused = kwargs.get("error_unused", False)
        output_types = ["clean", "doc", "original"]
        if self.output_type not in output_types:
            raise ValueError(
                f"Output type ({self.output_type}) must be one of {output_types}"
            )
        self.kwargs = kwargs

    @classmethod
    def parse_file(cls, file_name=None, additional_args=None, **kwargs):
        """
        Parse a file and/or string into a TomlParmParse object.

        Parameters
        ----------
           file_name: str (path-like), optional
              File to load as a tomlkit Document
           additional_args: str (toml), optional
              TOML format string of parameters to add to the file
           kwargs: optional
              Passed to TomlParmParse constructor

        Returns
        -------
           tpp: TomlParmParse
              A TOML ParmParser
        """
        if file_name is not None:
            try:
                with open(file_name) as tomlfile:
                    data = tomlkit.load(tomlfile)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"TomlParmParse: Input file <{file_name}> not found!"
                )
            name = f"<{file_name}>"
            if additional_args is not None:
                addl_data = tomlkit.loads(additional_args)
                data.update(addl_data)
                name = f"<{file_name}+cl_args>"

        elif additional_args is not None:
            data = tomlkit.loads(additional_args)
            name = "<cl_args>"
        else:
            raise RuntimeError(
                "TomlParmParse: must provide input file or arguments on command line"
            )

        return cls(data, name=name, is_base=True, **kwargs)

    @classmethod
    def parse_args(cls, description=None):
        """
        Parse command line arguments specifying file and arguments to create a TPP.

        Parameters
        ----------
           description: str, optional
              Short description of program for which config is being loaded

        Returns
        -------
           tpp: TomlParmParse
              A TOML ParmParser
        """
        if description is None:
            description = "A tool using the TomlParmParse class to parse inputs"
        description += (
            " --- This program uses the TomlParmParse utility to manage"
            "input/config files and command line arguments. Arguments can "
            "be auto-documented by enabling output of the 'doc' type."
        )
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "infile", nargs="?", default=None, help="Input file to parse inputs from"
        )
        parser.add_argument(
            "-o",
            "--output",
            default=None,
            help="File in which to write used inputs/outputs",
        )
        parser.add_argument(
            "-t",
            "--output_type",
            default="doc",
            help="Type of output file: *doc*: include comments documenting used inputs "
            "*clean*: no comments, only used inputs *original*: input file in "
            "original formatting",
        )
        parser.add_argument(
            "-a", "--args", default=None, help="Override arguments, as a toml string"
        )
        parser.add_argument(
            "-w",
            "--allow_overwrite",
            action="store_true",
            help="Disallow overwriting entries once they have been used or set",
        )
        parser.add_argument(
            "-e",
            "--error-unused",
            action="store_true",
            help="Raise error if there are unused inputs",
        )
        args = parser.parse_args()
        if args.args is not None:
            args.args = args.args.replace("\\n", "\n").replace(";", "\n")
        return cls.parse_file(
            args.infile,
            additional_args=args.args,
            output=args.output,
            output_type=args.output_type,
            no_overwrite=not args.allow_overwrite,
            error_unused=args.error_unused,
        )

    def __repr__(self):
        """Provide string represention of data (as nested dict)."""
        return self.data.__repr__()

    def __getitem__(self, item_name):
        """
        Provide string represention of data (as nested dict).
        """
        # Use periods to separate hierarchy levels in item_name.
        # if none, we're at the last requested level
        if item_name.count(".") == 0:
            if item_name in self.data:
                item = self.data[item_name]
                if hasattr(item, "keys") and hasattr(item, "values"):
                    # dict-like: return a sub-tomlparmparse object
                    if item_name not in self.accessed_data:
                        self.accessed_data[item_name] = tomlkit.document()
                    return TomlParmParse(
                        self.data[item_name],
                        self.accessed_data[item_name],
                        f"{self.name}.{item_name}",
                        **self.kwargs,
                    )
                else:
                    # we're at a leaf with a value to return
                    if item_name not in self.accessed_data:
                        self.accessed_data[item_name] = item
                    return item
            else:
                return None
        else:
            # split keyword on the first period
            split_loc = item_name.index(".")
            prefix = item_name[:split_loc]
            suffix = item_name[split_loc + 1 :]
            subtpp = self[prefix]
            if subtpp is not None:
                return subtpp[suffix]
            else:
                return subtpp

    def __setitem__(self, item_name, value):
        # Use periods to separate hierarchy levels in item_name
        # if none, we're at the last requested level
        if item_name.count(".") == 0:
            if (item_name in self.accessed_data) and self.no_overwrite:
                if value != self.accessed_data[item_name]:
                    raise ValueError(
                        "cannot set a new value for an already accessed "
                        f"item <{item_name}> in TomlParmParse object {self.name}"
                    )
            else:
                self.accessed_data[item_name] = value
                self.data[item_name] = self.accessed_data[item_name]
        else:
            # split keyword on the first period
            split_loc = item_name.index(".")
            prefix = item_name[:split_loc]
            suffix = item_name[split_loc + 1 :]
            if prefix not in self.data:
                self.data[prefix] = tomlkit.document()
            self[prefix][suffix] = value

    def get(self, item_name, default=None, doc=None):
        retval = self[item_name]
        if retval is None:
            if default is not None:
                retval = default
            else:
                raise RuntimeError(
                    f"Required item <{item_name}> (doc: {doc}) not found in"
                    f"TomlParmParse object {self.name}."
                )
        self[item_name] = retval

        if doc is not None and self.output_type == "doc":
            if default is not None:
                doc += f"  | optional, default: {default}"
            self[item_name].comment(doc)

        return retval

    def set(self, item_name, value, doc=None):
        self[item_name] = value
        if doc is not None and self.output_type != "clean":
            self[item_name].comment(doc)

    def doc(self, doc):
        self.data.comment(doc)
        self.accessed_data.comment(doc)
        return self

    def dump(self, outfile=None):
        if outfile is not None:
            write_to = outfile
        elif self.output is not None:
            write_to = self.output
        else:
            raise RuntimeError(
                "Cannot dump with no outfile unless initialized with one"
            )

        with open(write_to, "w") as tomlfile:
            if self.output_type == "original":
                tomlfile.write(tomlkit.dumps(self.data))
            else:
                tomlfile.write(tomlkit.dumps(self.accessed_data))

    def check_unused_inputs(self):
        """Return any items in table that have not been used."""
        unused = []
        for key in self.data:
            if key not in self.accessed_data:
                unused.append(key)
            else:
                item = self[key]
                if hasattr(item, "check_unused_inputs"):
                    unused_subkeys = item.check_unused_inputs()
                    unused += [f"{key}.{subkey}" for subkey in unused_subkeys]
        return unused

    def __del__(self):
        """When destroying, optionally dump output to file, raise error for unused."""
        if self.is_base:
            if self.output is not None:
                self.dump()
            unused = self.check_unused_inputs()
            if len(unused) > 0 and self.error_unused:
                print(f"Unused TomlParmParse inputs for {self.name}:")
                for key in unused:
                    print(f"    {key}")
                raise RuntimeError("Aborting due to unused inputs.")
