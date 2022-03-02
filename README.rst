vlne
====

Package to train Deep Learning neutrino energy estimators operating on particle
information.

This package is a continuation of the `lstm_ee <lstm_ee_>`_ work, but aimed at
supporting version 2 of TensorFlow framework.

Overview
--------

This package is designed to simplify training of the energy estimators for
various neutrino experiments. It contains an extensive library of helper
functions and a number of scripts to train different kinds of energy estimators
and evaluate them. Please refer to `Documentation`_ for the details.

This package is inspired by the `rnnNeutrinoEnergyEstimator <original_>`_
which laid the groundwork for the deep learning energy estimator development.

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

   python -m unittest


Requirements
------------

`vlne` package is written in python v3 and won't work with python v2.
`vlne` depends on the following packages:

- ``tensorflow`` -- for training of neural networks.
  TensorFlow v2.9 and above are tested. One may try running `vlne` with the
  lower TF versions, but you may need to modify some code parts. Unfortunately,
  TF does not guarantee backward compatibility even between minor releases.
- ``pandas``, ``numpy`` -- basis of python data handling.
- ``scipy``   -- for fitting curves.
- `vlndata <https://github.com/usert5432/vlndata>`_ -- for working with tabular
  and variable length array data formats. **Needs manual installation**.

Make sure that these packages are available on your system.

There are multiple ways these packages can be installed.
The `contrib/containers` subdirectory provides definitions of various
containers that pack the above mentioned packages. Alternatively, these
packages can be installed with ``pip`` by running

.. code-block:: bash

   pip install --user -r requirements.txt

Also, `vlne` has a number of optional dependencies:

* ``speval`` -- for parallelizing training across multiple machines.
* ``cafplot`` -- for creating and plotting histograms


Documentation
-------------

`vlne` package comes with a basic documentation in the sphinx format.
You can compile the sphinx documentation by running the following command in
the ``doc`` subdirectory (requires ``sphinx`` installed):

.. code-block:: bash

   make html

It will build all available documentation, which can be viewed with a web
browser by pointing it to the ``build/html/index.html`` file.

In addition to the sphinx documentation the `vlne` code is covered by a
numpy like docstrings. Please refer to the docstrings and the source code for
the details about inner `vlne` workings.


Example Usage
~~~~~~~~~~~~~

This subsection provides a summary of a usage example of the `vlne` package to
train and evaluate a Deep Learning energy estimator. To start training a
neutrino energy estimator one needs to obtain a training dataset.

The instructions on how to obtain a training dataset are experiment specific.
For example, NOvA's training dataset can be obtained according to following
`guide <doc/source/tutorials/nova/data.rst>`__, and
`this link <doc/source/tutorials/dune/data.rst>`__ gives instructions specific
to the DUNE experiment.

After the training dataset has been obtained, one can start the training
itself. `vlne` packs multitude of training scripts for various experiments
and training variations (under ``scripts/train``). All the scripts are
essentially declarative files that define a training configuration and call
``train(config)``.

The training configuration is hierarchical and has intuitive structure.
It can be easily modified for a particular need. One can use this
``scripts/train/nova/numu/mprod5/final/train_fd_fhc.py`` training script as a
starting point for developing a training configuration for a new dataset.

Finally, once the training is complete, one can begin evaluation of the
performance of the Deep Learning energy estimator. `vlne` has several
scripts to do that under the ``scripts/eval`` subdirectory. For example,
``scripts/eval/eval_model.py`` can be used to evaluate energy
resolution of the energy estimators.


.. _original: https://github.com/AlexanderRadovic/rnnNeutrinoEnergyEstimator
.. _lstm_ee: https://github.com/usert5432/lstm_ee

