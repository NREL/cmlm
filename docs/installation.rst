Installation
============

CMLM is just collection of python scripts, but some installation steps are required. It relies on several python libraries that you must have installed in order to be able to run the CMLM scripts. Most are fairly standard, but you can follow the directions below in order to ensure you have compatible versions of all relevant dependencies.

First, it is a good idea to use a tool like ``conda`` to create and activate an environment for your install, but that isn't strictly required::

  $ conda create -n cmlm python=3.11
  $ conda activate cmlm

Dependencies
------------

Dependencies for CMLM are primarily managed through `poetry <https://python-poetry.org/docs/#installation>`_. Poetry is therefore recommended to use CMLM and can typically be installed through system package managers (e.g. HomeBrew) or following the instructions in poetry's documentation.

To install all CMLM dependencies, from the base directory of the CMLM repository, simply run::

  $ poetry install --all-extras

Then, run any python script (replace ``script_name.py`` with the desired script) using::

  $ poetry run python script_name.py

Many users will not require or desire all dependencies. A base install with limited functionality can be achieved by omitting the `--all-extras`. Available
dependency groups include ``ml``, ``parallel``, and ``dev`` and can be specified using, for example ``poetry install --extras ml``. The ``parallel`` group should only be included if you have a valid MPI installation.

If you prefer, you can review the required dependencies in the ``pyproject.toml`` file and manually install with your favorite package manager, then run the python scripts as normal. We also support installation through pip (using the  ``[]`` to chose which sets of optional dependencies to include)::

  $ pip install -e .[ml,parallel,dev]

.. note::

   Currently only dependencies for the tabulated chemistry portion of CMLM are managed properly. Adding this support for the ML part of CMLM is a work in progress.
