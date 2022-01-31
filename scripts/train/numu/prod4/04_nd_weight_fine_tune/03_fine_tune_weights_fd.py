"""Train family of LSTM v3 FD networks with different weights"""

import os
from speval import speval

from vlne.args    import join_dicts
from vlne.consts  import ROOT_OUTDIR
from vlne.presets import PRESETS_TRAIN
from vlne.train   import create_and_train_model
from vlne.utils   import parse_concurrency_cmdargs, setup_logging

config = join_dicts(
    PRESETS_TRAIN['numu_v3'],
    {
    # Config:
        'batch_size'   : 1024,
        #'vars_input_slice',
        #'vars_input_png2d',
        #'vars_input_png3d',
        #'vars_target_total',
        #'vars_target_primary',
        'dataset'      : (
            'numu/prod4/fd_fhc'
            '/dataset_vlne_fd_fhc_nonswap_loose_cut.csv.xz'
        ),
        'early_stop'   : {
            'name'   : 'standard',
            'kwargs' : {
                'monitor'   : 'val_loss',
                'min_delta' : 0,
                'patience'  : 40,
            },
        },
        'epochs'       : 200,
        'loss'         : 'mean_absolute_percentage_error',
        'max_prongs'   : None,
        'model'        : {
            'name'   : 'lstm_v3',
            'kwargs' : {
                'norm'         : 'batch',
                'layers_pre'   : [ 128, 128, 128 ],
                'layers_post'  : [ 128, 128, 128 ],
                'lstm_units2d' : 32,
                'lstm_units3d' : 32,
                'n_resblocks'  : 0,
            },
        },
        'noise'        : None,
        'optimizer'    : {
            'name'   : 'RMSprop',
            'kwargs' : { 'lr' : 0.001 },
        },
        'prong_sorters' : None,
        'regularizer'   : {
            'name'   : 'l1',
            'kwargs' : { 'l' : 0.001 },
        },
        'schedule'     : {
            'name'   : 'standard',
            'kwargs' : {
                'monitor'  : 'val_loss',
                'factor'   : 0.5,
                'patience' : 5,
                'cooldown' : 0
            },
        },
        'seed'            : 1337,
        'steps_per_epoch' : 250,
        'test_size'       : 200000,
        'weights'         : None,
    # Args:
        'vars_mod_png2d'  : None,
        'vars_mod_png3d'  : None,
        'vars_mod_slice'  : None,
        'outdir'          : (
            'numu/prod4/04_nd_weights_fine_tune/'
            '03_weights_fine_tune_fd/'
        ),
    }
)

parse_concurrency_cmdargs(config)

search_space = [ { 'weights' : 'weight' }, ]

for bins in [ 7, 10, 20, 35, 70 ]:
    for clip in [ None, 3, 5, 10, 15, 25, 50, 75, 100 ]:
        search_space.append({
            'weights' : {
                'name'   : 'flat',
                'kwargs' : { 'bins' : bins, 'range' : (0, 7), 'clip' : clip },
            }
        })

for bins in [ 5, 10, 25, 50 ]:
    for clip in [ None, 3, 5, 10, 15, 25, 50, 75, 100 ]:
        search_space.append({
            'weights' : {
                'name'   : 'flat',
                'kwargs' : { 'bins' : bins, 'range' : (0, 5), 'clip' : clip },
            }
        })

logger = setup_logging(
    log_file = os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

speval(
    lambda x : create_and_train_model(**config, extra_kwargs = x),
    search_space,
    os.path.join(ROOT_OUTDIR, config['outdir'], "trials.db"),
    timeout = 3 * 60 * 60
)

