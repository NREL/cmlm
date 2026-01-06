"""A module for interacting with TOML format input files based on tomlkit."""

import argparse
import os
import sys
import textwrap

import tomlkit


def scalar_to_list(val):
    """Leave lists as is, convert scalars to 1 element lists."""
    if not isinstance(val, (list, tuple)):
        return [val]
    else:
        return val


def recursively_update_dict(base, new):
    """
    Update items and subitems in one nested dict-like object based on another.

    For each key in new.keys(), replace base[key] with new[key], unless
    new[key] also has a .keys() attribute, in which case this function
    is recursively called to update the sub-dict-like.

    Parameters
    ----------
        base: dict-like
            Input (nested) dictionary to update
        new: dict-like
            (Nested) dictionary entries to update in base
    """
    for key in new:
        if key not in base:
            base[key] = new[key]
        else:
            if hasattr(new[key], "keys"):
                if not hasattr(base[key], "keys"):
                    base[key] = {}
                recursively_update_dict(base[key], new[key])
            else:
                base[key] = new[key]


def beautify_document(doc, max_inline_length=20, max_line_length=88):
    """
    Move inline comments top line above item in TOML string.

    Parameters
    ----------
        doc: str (TOML format)
            string representation of document being updated
        max_inline_length: int, default 20
            Maximum length of comment to leave as inline, but total line
            length must still be less than max_line_length or comment will be moved
        max_line_length: int, default 88
            Maximum line length for wrapping comment text
    """
    fancy_line = "# " + "=" * (max_line_length - 4) + " #"
    lines = doc.split("\n")
    new_lines = []
    for line in lines:
        is_category = False
        if line.startswith("["):
            is_category = True
            if len(new_lines) != 0:
                if new_lines[-1] != "":
                    new_lines.append("")
                new_lines.append("")
            new_lines.append(fancy_line)

        if "#" in line:
            text, comment = line.split("#", 1)
            if (
                len(comment) > max_inline_length
                or len(line) > max_line_length
                or is_category
            ):
                comment_lines = textwrap.wrap(comment, max_line_length - 2)
                if len(new_lines) != 0:
                    if new_lines[-1] != "" and not is_category:
                        new_lines.append("")  # newline
                for cl in comment_lines:
                    if cl.startswith(" "):
                        cl = cl[1:]
                    new_lines.append("# " + cl)
                if is_category:
                    new_lines.append(fancy_line)
                new_lines.append(text)
                new_lines.append("")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


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
        base: TomlParmParse, optional
            Parent TomlParmParse object. If None, this is a base TomlParmParse
            object and output will occur during garbage collecting if requested.
            Default None.
        output: string, optional kwarg
            Directory in which to save output. Default None (no output saved).
        output_type: str, optional kwarg
            Type of output to save: "clean" will save only inputs used with no comments.
            "doc" will save only inputs used with comments generated based on doc info
            provided when accessing variables. "original" keeps all variables, comments,
            and formatting from the provided input file. Default "clean".
        no_overwrite: bool, optional kwarg
            Raise an error if multiple different values are set/accessed for a variable.
            Default False.
        error_unused: bool, optional kwarg
            Raise an error for unused variables in input file. Default False.
        live_update: bool, optional kwarg
            Continuously update output file as code runs. Default False.
    """

    def __init__(
        self, datadict, accessed_data=None, name="<base>", base=None, **kwargs
    ):
        self.data = datadict

        if accessed_data is not None:
            self.accessed_data = accessed_data
        else:
            accessed_data = tomlkit.document()
            self.accessed_data = accessed_data

        self.name = name
        self.base = base
        self.output = kwargs.get("output", None)
        if self.output is not None:
            # if output ends with .toml, its a file, otherwise its a directory
            if not self.output.endswith(".toml"):
                self.output = os.path.join(self.output, "config.toml")
            self.output_dir = os.path.split(self.output)[0]
            if self.base is None:
                if len(self.output_dir) > 0 and not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
        self.output_type = kwargs.get("output_type", "clean")
        self.no_overwrite = kwargs.get("no_overwrite", False)
        self.error_unused = kwargs.get("error_unused", False)
        self.live_update = kwargs.get("live_update", False)
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
                recursively_update_dict(data, addl_data)
                name = f"<{file_name}+cl_args>"

        elif additional_args is not None:
            data = tomlkit.loads(additional_args)
            name = "<cl_args>"
        else:
            raise RuntimeError(
                "TomlParmParse: must provide input file or arguments on command line"
            )

        return cls(data, name=name, base=None, **kwargs)

    @classmethod
    def parse_args(cls, description=None, infile=None, require_output=False):
        """
        Parse command line arguments specifying file and arguments to create a TPP.

        Parameters
        ----------
           description: str, optional
              Short description of program for which config is being loaded
           infile: str, optional
              Default TOML input file to use
           require_output: bool, optional
              if true, the `-o` command line argument is required. Default False.

        Returns
        -------
           tpp: TomlParmParse
              A TOML ParmParser
        """
        if description is None:
            description = "A tool using the TomlParmParse class to parse inputs"
        description += (
            " --- This program uses the TomlParmParse utility to manage "
            "input/config files and command line arguments. Arguments can "
            "be auto-documented by enabling output of the 'doc' type."
        )
        if infile is None:
            infile = sys.argv[0].replace(".py", ".toml")
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "infile", nargs="?", default=infile, help="Input file to parse inputs from"
        )
        parser.add_argument(
            "-o",
            "--output",
            default=None,
            required=require_output,
            help="File (.toml) or directory in which to write used inputs/outputs",
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
            "--no_overwrite",
            action="store_true",
            help="Disallow overwriting entries once they have been used or set",
        )
        parser.add_argument(
            "-e",
            "--error_unused",
            action="store_true",
            help="Raise error if there are unused inputs",
        )
        parser.add_argument(
            "-l",
            "--live_update",
            action="store_true",
            help="Update output file continuously as code runs",
        )
        args = parser.parse_args()
        if args.args is not None:
            args.args = args.args.replace("\\n", "\n").replace(";", "\n")
        return cls.parse_file(
            args.infile,
            additional_args=args.args,
            output=args.output,
            output_type=args.output_type,
            no_overwrite=args.no_overwrite,
            error_unused=args.error_unused,
            live_update=args.live_update,
        )

    def __repr__(self):
        """Provide string representation of data (as nested dict)."""
        return self.data.__repr__()

    def __getitem__(self, item_name):
        """Return a sub-TomlParmParse object, or value if requesting a leaf."""
        # Use periods to separate hierarchy levels in item_name.
        # if none, we're at the last requested level
        if item_name.count(".") == 0:
            if item_name in self.data:
                item = self.data.item(item_name)
                if hasattr(item, "keys") and hasattr(item, "values"):
                    # dict-like: return a sub-tomlparmparse object
                    if item_name not in self.accessed_data:
                        self.accessed_data[item_name] = tomlkit.document()
                    return TomlParmParse(
                        self.data[item_name],
                        self.accessed_data[item_name],
                        f"{self.name}.{item_name}",
                        base=self,
                        **self.kwargs,
                    )
                else:
                    # we're at a leaf with a value to return
                    if item_name not in self.accessed_data:
                        self.accessed_data[item_name] = item
                        if self.output_type == "clean":
                            item.trivia.comment = ""
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
        """Set the value of a leaf in the table hierarchy."""
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
                self.data[item_name] = self.accessed_data.item(item_name)
                if self.live_update and self.output is not None:
                    self.dump()
        else:
            # split keyword on the first period
            split_loc = item_name.index(".")
            prefix = item_name[:split_loc]
            suffix = item_name[split_loc + 1 :]
            if prefix not in self.data:
                self.data[prefix] = tomlkit.document()
            self[prefix][suffix] = value

    def get(self, item_name, default=None, doc=None, choices=None):
        """
        Retrieve a leaf or subtable form the TomlParmParse table.

        Can optionally specify a default, otherwise an error will be raised if
        not found. Can also optionally specify a doc string to add to the
        TOML file for the input being retrieved.

        Parameters
        ----------
            item_name: str
                key to lookup in table. May parse through multiple levels in hierarchy
                by separating different levels with periods, e.g. lev1name.level2name.lev3name
            default: optional
                value to use if item_name is not found in table
            doc: optional
                string to add as a comment in the TOML file
            choices: optional
                list or tuple of allowable options for input parameter. If specified,
                an error will be raised if the specified value is not in the list.

        Returns
        -------
            retval: TomlParmParse or any
                If item_name is not a leaf, a subtable of the TomlParmParse object, otherwise
                the leaf (which may be a string, int, list, etc.)
        """
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

        if choices is not None:
            if retval not in choices:
                raise ValueError(
                    f"In TomlParmParse object {self.name}:\n"
                    f"  Invalid value specified for item <{item_name}> (doc: {doc})\n"
                    f"  Choices are: {choices}"
                )

        if doc is not None and self.output_type == "doc":
            if default is not None:
                doc += f"  | optional, default: {default}"
            if choices is not None:
                doc += f"  | choices are: {choices}"
            self[item_name].comment(doc)
            if self.live_update and self.output is not None:
                self.dump()

        return retval

    def set(self, item_name, value, doc=None):
        """
        Set value for a leaf in the table.

        Parameters
        ----------
            item_name: str
               key for the item being added
            value: int, str, list, etc.
               value for the item being added (must be TOML format compatible type)
            doc: str
               comment to add for item in TOML file
        """
        self[item_name] = value
        if doc is not None and self.output_type != "clean":
            self[item_name].comment(doc)

    def doc(self, doc):
        """
        Add a high level comment to the TOML document.

        Parameters
        ----------
            doc: optional
                string to add as a comment in the TOML file

        Returns
        -------
            self: TomlParmParse
                The updated TomlParmParse object
        """
        if self.output_type == "doc":
            self.data.comment(doc)
            self.accessed_data.comment(doc)
        return self

    def dump(self, outfile=None):
        """
        Output TomlParmParse data to file.

        Parameters
        ----------
            outfile: str, optional
                file name to save to
        """
        if self.base is None:
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
                elif self.output_type == "doc":
                    tomlfile.write(beautify_document(tomlkit.dumps(self.accessed_data)))
                else:
                    tomlfile.write(tomlkit.dumps(self.accessed_data))
        else:
            self.base.dump(outfile)

    def check_unused_inputs(self):
        """Return any keys in table that have not been used."""
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
        if self.base is None:
            if self.output is not None:
                self.dump()
            unused = self.check_unused_inputs()
            if len(unused) > 0 and self.error_unused:
                print(f"Unused TomlParmParse inputs for {self.name}:")
                for key in unused:
                    print(f"    {key}")
                raise RuntimeError("Aborting due to unused inputs.")
