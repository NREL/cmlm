.. _getting-started:

Getting Started
===============

CMLM is just a python package and a collection of scripts that utilize that package. It depends on many standard python libraries and the workflow for installing and using it resembles many python codes for scientific computing.

Dependencies & Installation
---------------------------

First, it is a good idea to use a tool like ``conda`` to create and activate an environment for your install, but that isn't strictly required::

  $ conda create -n cmlm python=3.11
  $ conda activate cmlm

Dependencies for CMLM are primarily managed through `poetry <https://python-poetry.org/docs/#installation>`_. Poetry is therefore recommended to use CMLM and can be installed in a variety of ways. Most recommended would be to install it within the environment you just created (``$ conda install poetry``) but it may also be installed through system package managers (e.g. HomeBrew) or following the instructions in poetry's documentation.

To install all CMLM dependencies, from the base directory of the CMLM repository, simply run::

  $ poetry install --all-extras

Then, run any python script (replace ``script_name.py`` with the desired script) using::

  $ poetry run python script_name.py

Many users will not require or desire all dependencies. A base install with limited functionality can be achieved by omitting the ``--all-extras``. Available
dependency groups include ``ml``, ``parallel``, and ``dev`` and can be specified using, for example ``poetry install --extras ml``. The ``parallel`` group should only be included if you have a valid MPI installation.

If you prefer, you can review the required dependencies in the ``pyproject.toml`` file and manually install with your favorite package manager, then run the python scripts as normal. We also support installation through pip (using the  ``[]`` to chose which sets of optional dependencies to include)::

  $ pip install -e .[ml,parallel,dev]

.. note::

   Currently only dependencies for the tabulated chemistry portion of CMLM are managed properly. Adding this support for the ML part of CMLM is a work in progress.

Using CMLM
----------

CMLM contains three components:

- :ref:`library-api`: The core of CMLM is a python package that contains classes and functions associated with generating tabulated models and training neural network models. Corresponds to the ``cmlm/`` subdirectory in the repository.

- :ref:`command-line-tools`: Runable utilities installed as part of the CMLM package for interacting with the tabulated and neural network models. Also comes from the ``cmlm/`` subdirectory in the repository.

- :ref:`run-scripts`: A set of sample scripts that utilize the API above for various aspects of the model generation workflow, including data generation, data processing, and synthesizing data into a model. Corresponds to the ``run-scripts`` subdirectory in the repository.

Most users will begin by running the contained example scripts (supplemented by the command line tools) to generate models, then incorporate these models into simulations performed with the Pele suite. Advanced users will create their own scripts utilizing the CMLM package through the documented API to create new types of models.

Head to the :ref:`tutorials` section for example workflows that leverage CMLM to generate reduced-order manifold turbulent combustion models and then apply these in Pele simulations.
