.. _command-line-tools:

Command Line Tools
==================

ctable_tools
-------------

The ``ctable_tools`` script that gets installed as part of cmlm has several features for plotting, printing, slicing, etc. To see what the available features are and how to use them, run::

  poetry run ctable_tools

assuming you have installed ``cmlm`` with poetry, which generates the following list
of options

.. code-block:: console

   usage: ctable_tools [-h] [-f {NGA,Pele}] [-d DIMENSIONS]
                       [-dl DIMENSION_LABELS [DIMENSION_LABELS ...]] [-p] [-o OUTPUTFILE]
                       [-cf] [-cu {none,cgs2mks,mks2cgs}] [-sl SLICE [SLICE ...]]
                       [-sp SLICE_PLOT [SLICE_PLOT ...]] [-v VARIABLES [VARIABLES ...]]
                       inputfile

   Useful tools for interacting with chemtable files

   positional arguments:
     inputfile             Table file to read

   options:
     -h, --help            show this help message and exit
     -f {NGA,Pele}, --format {NGA,Pele}
                           Table format, NGA or Pele
     -d DIMENSIONS, --dimensions DIMENSIONS
                           Number of dimensions in NGA table
     -dl DIMENSION_LABELS [DIMENSION_LABELS ...], --dimension_labels DIMENSION_LABELS [DIMENSION_LABELS ...]
                           List of dimension names in NGA table
     -p, --print           Flag to print extrema of table file
     -o OUTPUTFILE, --outputfile OUTPUTFILE
                           Output file to write, if needed
     -cf, --convert_format
                           Convert format from input format to other format
     -cu {none,cgs2mks,mks2cgs}, --convert_units {none,cgs2mks,mks2cgs}
                           Convert units and save new table
     -sl SLICE [SLICE ...], --slice SLICE [SLICE ...]
                           Create a table by slicing. Specify dimensions to slice and
                           values as a list of form dim_name1:value dim_name2:value
     -sp SLICE_PLOT [SLICE_PLOT ...], --slice_plot SLICE_PLOT [SLICE_PLOT ...]
                           Plot a slice of the table. Specify dimensions to slice and
                           values as a list of form dim_name1:value dim_name2:value, must
                           specify enough slice dims such that there are exactly one or
                           two slice dimensions. Alternatively, set as 'justplot' to
                           directly plot a 1D or 2D table without slicing.
     -v VARIABLES [VARIABLES ...], --variables VARIABLES [VARIABLES ...]
                           variables to be plottedif unspecified, all variables in table
                           will be plotted
