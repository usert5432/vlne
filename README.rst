vlne
====

Package to train Deep Learning neutrino energy estimators operating on
information particle information.

Overview
--------
This package is designed to simplify training of the energy estimators for
neutrino experiments. It contains an extensive library of helper functions and
a number of scripts to train various kinds of energy estimators and evaluate
them. Please refer to `Documentation`_ for the details.

This package is inspired by the `rnnNeutrinoEnergyEstimator <original_>`_
which laid the groundwork for the LSTM energy estimator development.

Installation
------------
`vlne` package is intended for developers. Therefore, it is recommended to
install the live version of the package, for example:

1. Git clone this repository:

.. code-block:: bash

   git clone https://github.com/usert5432/vlne

2. Setup `vlne` package, by running

.. code-block:: bash

   python setup.py develop --user

in the `vlne` directory.

If you are running `vlne` for the first time it might be useful to run
its test suite to make sure that the package is not broken:

.. code-block:: bash

   python -m unittest tests.run_tests.suite


Requirements
------------

`vlne` package is written in python v3 and won't work with python v2.
`vlne` depends on the following packages:

* ``keras``   -- for training neural networks.
* ``pandas``, ``numpy`` -- for handling data.
* ``cython``  -- for compiling optimized data handling functions
* ``scipy``   -- for fitting curves.
* ``cafplot`` -- for plotting evaluation results.

Make sure that these packages are available on your system. You can install
them with ``pip`` by running

.. code-block:: bash

   pip install --user -r requirements.txt

Also, `vlne` has a number of optional dependencies:

* ``tensorflow`` -- for exporting ``keras`` models into protobuf format that
  NOvASoft expects. Note that only ``tensorflow`` v1 is supported currently.

* ``pytables`` -- for working with HDF5 files.
* ``speval`` -- for parallelizing training across multiple machines.


Documentation
-------------

`vlne` package comes with several layers of documentation. The basic
overview of the `vlne` workings and examples of usage are documented in
sphinx format. You can find this documentation by the following
`link <prebuilt_doc_>`_ (requires nova credentials).

Alternatively, you can manually compile the sphinx documentation by running
the following command in the ``doc`` subdirectory (requires ``sphinx``
installed):

.. code-block:: bash

   make html

It will build all available documentation, which can be viewed with a web
browser by pointing it to the ``build/html/index.html`` file.

In addition to the sphinx documentation the `vlne` code is covered by a
numpy like docstrings. Please refer to the docstrings and the source code for
the details about inner `vlne` workings.

.. _prebuilt_doc: https://nova-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=45821
.. _original: https://github.com/AlexanderRadovic/rnnNeutrinoEnergyEstimator

