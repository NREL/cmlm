Installation
============

CMLM is a collection of python scripts, so no true installation is necessary. However, it does rely on several python libraries that you must have installed in order to be able to run the CMLM scripts. Most are fairly standard, but you can follow the directions below in order to ensure you have compatible versions of all relevant dependencies.

Dependencies
------------

Dependencies for CMLM are managed through `poetry <https://python-poetry.org/docs/#installation>`_. Poetry is therefore recommended to use CMLM and can typically be installed through system package managers (e.g. HomeBrew) or following the instructions in poetry's documentation.

To install CMLM dependencies, from the base directory of the CMLM repository, simply run::

  $ poetry update

Then, run any python script (replace ``script_name.py`` with the desired script) using::

  $ poetry run python script_name.py

If you prefer, you can review the required dependencies in the ``pyproject.toml`` file and manually install with your favorite package manager, then run the python scripts as normal.

.. note::

   Currently only dependencies for the tabulated chemistry portion of CMLM are managed through ``poetry``. Adding this support for the ML part of CMLM is a work in progress.
