Questions & Contributing
========================

Discussions
-----------

If you have any questions about using the CMLM software, please create a post on the `GitHub Discussions <https://github.com/NREL/cmlm/discussions>`_ page.

Contributions
-------------

User contributions to the code are welcome! Please `open a Pull Request <https://github.com/NREL/cmlm/pulls>`_ with your proposed new feature, bug fix, documentation, etc.

Formatting Guidelines
~~~~~~~~~~~~~~~~~~~~~

CMLM strives to follow a rigid code style, managed through the ``black``, ``isort``, and ``flake8`` tools. The tools necessary to format Python code (currently only used within CEPTR) are maintained through Poetry.

1) Run ``isort``::

    poetry run isort .

This will perform proper sorting of the installed Python libraries.

2) Run ``black``::

    poetry run black .

This will perform proper formatting of all Python files to be consistent with all current files.

3) Run ``flake8``::

    poetry run flake8 .

This will run diagnostics on all the Python files and will list a series of issues that need to be addressed to adhere to current Python best practices.


Once all ``flake8`` messages have been addressed, the code will match the CMLM standard.

.. note::

   Consistent application of these guidelines is a work in progress. Old code is being modified to match them over time; these guidelines will be required for all new contributions.

Building the Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Thisi wordis not spelled correctly.

Please document any new features! After adding some documentation, you can do a test rebuild of the HTML documentation pages by going to the ``docs/`` directory and running ::

    poetry run make html
