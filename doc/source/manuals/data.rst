Data Handling
=============

This page will try to document data formats that `vlne` employs, data
generation performance and ways to improve it.

Data Formats
------------

Currently, `vlne` supports reading data from ``csv`` and ``hdf5`` files.

CSV Files
^^^^^^^^^

`vlne` package support reading data from the ``csv`` files. It supports
reading both plain files and compressed (*gzip*, *xz*, etc).

To store particle level variables (i.e. variable length arrays) in the ``csv``
files they are serialized as strings. For example

::

    "0.2,0.334,0.564,1.4"

will correspond to a variable length array

::

    [ 0.2, 0.334, 0.564, 1.4 ]

HDF5 Files
^^^^^^^^^^

`vlne` also supports reading data from the ``hdf5`` files. However, the
``hdf5`` files are expected to have a certain structure. Namely, the input
variables are expected to be stored as (N, ) arrays in the root of the
file. One array per variable. Variable length array (``vlarray``) variables
should be stored as arrays of ``vlarray``. For example, a valid file can have
the following structure:

::

    test.h5
        /calE, type earray, shape (N,)
        /nHit, type earray, shape (N,)
        ...
        /png.calE, type vlarray, shape (N,)
        /png.nhit, type vlarray, shape (N,)
        ...

Here ``/calE``, ``/hHit`` are datasets of event level calorimetric energies
and numbers of hits. And ``/png.calE``, ``/png.hHit`` are the datasets of
particle level calorimetric energies and numbers of hits. All datasets in the
``hdf5`` file should have the same length.

.. note::
    You can convert a ``csv`` file to an ``hdf5`` file by using a script
    ``scripts/data/csv_to_hdf.py``.

Data Generation Performance
---------------------------

You may find that the usage of the above data formats is rather slow. To
speed up the data loading, `vlne` package provides in-memory caching mechanism.
To activate caching, simply add ``--cache`` option to the command line
arguments of the training script.

You may also want to enable ``--precache`` option, that will preload the
dataset into RAM before the training. Unlike normal caching, the preloading is
heavily parallelized and fast.

