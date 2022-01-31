import os

from lstm_ee.args    import join_dicts
from lstm_ee.consts  import ROOT_OUTDIR
from lstm_ee.presets import PRESETS_TRAIN
from lstm_ee.train   import create_and_train_model
from lstm_ee.utils   import parse_concurrency_cmdargs, setup_logging

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
            'numu/mprod5/fd_fhc'
            '/dataset_lstm_ee_fd_fhc_nonswap_loose_cut.csv.xz'
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
            'name'   : 'trans_v1',
            'kwargs' : {
                'batchnorm'      : True,
                'layers_post'    : [ 128, 128, 128 ],
                'input_shape'    : 128,
                'num_hidden'     : 128,
                'ffn_num_hidden' : 128,
                'num_heads'      : 8,
                'num_blocks'     : 3,
            },
        },
        'noise'        : {
            'noise'               : 'discrete',
            'noise_kwargs'        : { 'values' : [ -0.2, 0, 0.2 ] },
            'affected_vars_slice' : [ 'calE', 'orphCalE', 'remPngCalE' ],
            'affected_vars_png2d' : [ 'png2d.calE', 'png2d.weightedCalE' ],
            'affected_vars_png3d' : [
                'png.calE',
                'png.weightedCalE',
                'png.bpf[0].overlapE',
                'png.bpf[1].overlapE',
                'png.bpf[2].overlapE',
            ],
        },
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
        'test_size'       : 200000,
        'weights' : {
            'name'   : 'flat',
            'kwargs' : { 'bins' : 35, 'range' : (0, 7), 'clip' : 50 },
        },
    # Args:
        'vars_mod_png2d'  : None,
        'vars_mod_png3d'  : None,
        'vars_mod_slice'  : None,
        'outdir'          : 'numu/mprod5/final/fd_fhc',
    }
)


setup_logging(
    log_file = os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

parse_concurrency_cmdargs(config)
create_and_train_model(**config)

