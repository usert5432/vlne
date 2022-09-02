"""Convert custom CSV file into HDF file for `vlne` training"""

import argparse

import h5py
import tqdm
import numpy as np

from vlndata.data_frame import CSVMemFrame

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser("Convert CSV to HDF")

    parser.add_argument(
        'input',
        help    = 'Input HDF file',
        metavar = 'input',
        type    = str,
        nargs   = '+'
    )

    parser.add_argument(
        '-o', '--output',
        help     = 'Output File',
        type     = str,
        required = True
    )

    return parser

class HDFExporter():
    """Object that saves `IDataLoader` variables into HDF file.

    Parameters
    ----------
    path : str
        Name of the HDF output file.
    """

    def __init__(self, path):
        self._path  = path
        self._filters = {
            'compression'      : 'gzip',
            'compression_opts' : 9,
            'fletcher32'       : True,
        }
        self._f = h5py.File(path, 'w')

    def _export_scalar_column(self, column, values):
        hdf_dset = self._f.get(column)

        if hdf_dset is None:
            hdf_dset = self._f.create_dataset(
                column, data = values, maxshape = (None, ), chunks = True,
                **self._filters
            )
        else:
            hdf_dset.resize((len(hdf_dset) + len(values)), axis = 0)
            hdf_dset[-len(values):] = values

    def _export_vlarr_column(self, frame, column):
        hdf_dset = self._f.get(column)
        vtype    = h5py.special_dtype(vlen = frame.dtype)

        n = len(frame)
        values = []

        for i in tqdm.tqdm(range(n), desc = column, total = n):
            values.append(frame.get_vlarr(column, i))

        if hdf_dset is None:
            hdf_dset = self._f.create_dataset(
                column, data = values, dtype = vtype, maxshape = (None, ),
                chunks = True
            )
        else:
            hdf_dset.resize((len(hdf_dset) + len(values)), axis = 0)
            hdf_dset[-len(values):][:] = values

    def export(self, frame):
        for column in frame.columns():
            print(f"        {column}")
            values = frame[column]

            if np.issubdtype(values.dtype, np.number):
                self._export_scalar_column(column, values)
            else:
                self._export_vlarr_column(frame, column)

    def __del__(self):
        self._f.close()

def main():
    parser  = create_parser()
    cmdargs = parser.parse_args()

    exporter = HDFExporter(cmdargs.output)

    for idx,path in enumerate(cmdargs.input):
        print("Processing file %d of %d" % (idx + 1, len(cmdargs.input)))
        print("   Loading...")
        frame = CSVMemFrame(path)
        print("   Exporting...")
        exporter.export(frame)

    print("Done")

if __name__ == '__main__':
    main()

