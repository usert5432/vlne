"""
Constants that are widely used in vlne
"""

import os

DEF_SEED  = 1337
DEF_MASK  = 0.

LABEL_TOTAL     = 'total'
LABEL_PRIMARY   = 'primary'
LABEL_SECONDARY = 'secondary'

if 'VLNE_DATADIR' in os.environ:
    ROOT_DATADIR = os.environ['VLNE_DATADIR']
else:
    ROOT_DATADIR = '/'

if 'VLNE_OUTDIR' in os.environ:
    ROOT_OUTDIR = os.environ['VLNE_OUTDIR']
else:
    ROOT_OUTDIR = '/'

