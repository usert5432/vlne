"""Hybrid NuMu/NuE CC peak only training"""

import os
from speval import speval

from vlne.args    import join_dicts
from vlne.consts  import ROOT_OUTDIR
from vlne.presets import PRESETS_TRAIN
from vlne.train   import create_and_train_model
from vlne.utils   import parse_concurrency_cmdargs, setup_logging

config = join_dicts(
    PRESETS_TRAIN['uboone_numu_v1'],
    {
    # Config:
        'batch_size'   : 1024,
        #'vars_input_slice',
        #'vars_input_png2d',
        #'vars_input_png3d',
        #'vars_target_total',
        #'vars_target_primary',
        'dataset'      : (
            'uboone/2021-10-22_hybrid/'
            + 'dataset_train_run1+run2_numu+nue-shuffled_peak_only.hdf'
        ),
        'early_stop'   : {
            'name'   : 'standard',
            'kwargs' : {
                'monitor'   : 'val_loss',
                'min_delta' : 0,
                'patience'  : 40,
            },
        },
        'epochs'       : 1000,
        'loss'         : 'mean_absolute_percentage_error',
        'max_prongs'   : None,
        'model'        : {
            'name'   : 'lstm_v2',
            'kwargs' : {
                'norm'        : 'batch',
                'layers_pre'  : [ 128, 128, 128 ],
                'layers_post' : [ 128, 128, 128 ],
                'lstm_units'  : 32,
                'n_resblocks' : 0,
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
        'seed'            : 0,
        'shuffle_data'    : False,
        'steps_per_epoch' : 500,
        'test_size'       : 200000,
        'weights'         : None,
    # Args:
        'vars_mod_png2d'  : None,
        'vars_mod_png3d'  : None,
        'vars_mod_slice'  : [
            '+config.flavor.numu',
            '+config.flavor.nue',
            '+config.contain.full',
            '+event.nue_score',
            '+event.numu_score',
        ],
        'outdir' : 'uboone/hybrid/04_peak_training',
    }
)

parse_concurrency_cmdargs(config)

search_space = [
    {
        'weights' : None,
        'label'   : 'no_weight',
    },
]

for bins in [ 5, 10 ]:
    search_space.append({
        'label'   : f'weights_{bins}',
        'weights' : {
            'name'   : 'flat',
            'kwargs' : {
                'bins' : bins, 'range' : (0.7, 1.7), 'var' : 'truth.nuE',
            },
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

