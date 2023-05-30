from vlne.train import create_and_train_model

config = {
    "batch_size" : 1024,
    "data"       : {
        "extra_vars" : {
            "flat_weight": {
                "name"   : "flat",
                "kwargs" : {
                    "bins"  : 50,
                    "clip"  : 50,
                    "range" : [ 0, 5 ],
                    "var"   : "truth.nuE"
                },
            }
        },
        "frame" : {
            "name" : "hdf-ra-frame",
            "path" : (
                "uboone/mcc9_v0/dataset_train_fc+pc_numu+nue_shuffled.hdf"
            ),
        },
        "input_groups_scalar" : {
            "input_event" : [
                "config.contain.full",
                "config.flavor.numu",
                "config.flavor.nue"
            ]
        },
        "input_groups_vlarr" : {
            "input_vlarr" : [
                "particle.start.x",
                "particle.start.y",
                "particle.start.z",
                "particle.end.x",
                "particle.end.y",
                "particle.end.z",
                "particle.pdg.electron",
                "particle.pdg.gamma",
                "particle.pdg.muon",
                "particle.pdg.neutron",
                "particle.pdg.pion",
                "particle.pdg.pizero",
                "particle.pdg.proton",
                "particle.startMomentum.t",
                "particle.startMomentum.x",
                "particle.startMomentum.y",
                "particle.startMomentum.z"
            ]
        },
        "target_groups" : {
            "primary" : [ "truth.lepE", ],
            "total"   : [ "truth.nuE", ]
        },
        "seed"            : 0,
        "shuffle"         : False,
        "val_size"        : 0.2,
        "test_size"       : None,
        "transform_train" : [
            {
                'name'        : 'vlarr-shuffle',
                'vlarr_group' : 'input_vlarr',
            },
            "mask-nan"
        ],
        "transform_test"  : [ "mask-nan", ],
        "vlarr_limits"    : None,
        "weights" : {
            #"primary" : "flat_weight",
            "total"   : "flat_weight",
        },
    },
    'early_stop'   : {
        'name'      : 'standard',
        'monitor'   : 'val_loss',
        'min_delta' : 0,
        'patience'  : 40,
    },
    'epochs'       : 200,
    'loss'         : 'mean_absolute_percentage_error',
    'model'        : {
        'name'   : 'lstm_v2',
        'kwargs' : {
            'norm'        : 'batch',
            'seq_norm'    : 'batch',
            'layers_pre'  : [ 128, 128, 128 ],
            'layers_post' : [ 128, 128, 128 ],
            'lstm_units'  : 32,
            'n_resblocks' : 0,
        },
    },
    'optimizer'    : {
        'name' : 'RMSprop',
        'lr'   : 0.001,
    },
    'regularizer'   : {
        'name' : 'l2',
        'l'    : 0.008,
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
    'steps_per_epoch' : 500,
    # Args:
    'outdir'    : 'uboone/mcc9/final_v0',
    'log_level' : 'INFO',
}

create_and_train_model(**config)

