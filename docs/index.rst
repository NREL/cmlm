Welcome to the documentation for Co-optimized Machine-Learned Manifolds (CMLM)!
===============================================================================

CMLM includes scripts for generating and testing machine learned manifold models
for LES of turbulent combustion that use neural networks. The included machine
learning approaches were initially published in a `Combustion and Flame article <https://doi.org/10.1016/j.combustflame.2022.112286>`_.
It also includes scripts for generating physics-based manifold models with
tabulated chemistry data.
This package generates models in a format usable by the `PeleLMeX <https://amrex-combustion.github.io/PeleLMeX/>`_ reacting flow solver, with much of the implementation done through the `PelePhysics <https://github.com/AMReX-Combustion/PelePhysics>`_ library.

Head over to the :ref:`getting-started` page to see how to install and begin using CMLM.

Source code available on `GitHub <https://github.com/NREL/cmlm>`_.
**The repository and this documentation are a work in progress.** Adding this documentation is the
first step toward making it a usable piece of software with some semblance of
organization, testing, dependency management, etc.

.. warning::

   For now, CMLM is under active development. Changes to interfaces, file formats,
   outputs, behavior, etc. should all be expected and may be made without warning.

.. toctree::
   :maxdepth: 3
   :includehidden:
   :caption: Contents:

   getting-started
   library-api
   command-line-tools
   run-scripts
   tutorials
   contributing
   citation

.. footbibliography::
