Using CMLM
==========

CMLM contains separate components: a library/API with support for both tabulated and neural network reduced-order manifold chemistyr models, and some handy-dandy scripts for interacting with data files.

Library/API
-----------

TabulatedFunction
~~~~~~~~~~~~~~~~~

The ``TabulatedFunction`` capability derives from a pandas MultiIndex FataFrame, with added capability for interpolation and for reading/writing binary files in the format expected by the Pele codes. Any data that exists as a pandas MultiIndex DataFrame can easily by converted into a
tabulated function.

More documentation on the API to come later.

Machine Learning
~~~~~~~~~~~~~~~~

TODOC.

Helpful Scripts
---------------

Data Generation/Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

TODOC.

Interacting with Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODOC.

Interacting with Tabulated Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ctable_tools`` script that gets installed as part of cmlm has several features for plotting, printing, slicing, etc. To see what the available features are and how to use them, run::

  poetry run ctable_tools

assuming you have installed ``cmlm`` with poetry.
