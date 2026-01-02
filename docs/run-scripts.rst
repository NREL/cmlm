.. _run-scripts:

Run Scripts
===========

These scripts all use the TomlParmParse utility to manage configutation/inputs. Sample
inputs with descriptions are included in ``.toml`` files with the same base file name
as the script. All scripts can be run using::

  python script_name.py script_name.toml

Adding the ``-h`` flag will give the additional command line inputs that are associated
with the TomlParmParse utility::

  positional arguments:
    infile                Input file to parse inputs from

  options:
    -h, --help            show this help message and exit
    -o OUTPUT, --output OUTPUT
                          File (.toml) or directory in which to write used inputs/outputs
    -t OUTPUT_TYPE, --output_type OUTPUT_TYPE
                          Type of output file: *doc*: include comments documenting used inputs *clean*: no comments, only used inputs *original*: input file in original formatting
    -a ARGS, --args ARGS  Override arguments, as a toml string
    -w, --no_overwrite    Disallow overwriting entries once they have been used or set
    -e, --error_unused    Raise error if there are unused inputs
    -l, --live_update     Update output file continuously as code runs

Notably, specifying ``-o`` will result in a new toml file with the configuration that
was used, including documentation for each input by default, and ``-a`` allows inputs
that are usually specified in the input file to be specified on the command line.

----

cantera
-------

These python scripts use Cantera to generate 1D flame solutions that will then be
compiled into tabulated chemistry tables.

----

compute_premixed_flamelets.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule::
   compute_premixed_flamelets

----

ctable
------

These python scripts use the CMLM package to generate tabulated chemistry data.

----

create_dummy_table_nd.py
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule::
   create_dummy_table_nd

----

create_spray_table_nd.py
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule::
   create_spray_table_nd
