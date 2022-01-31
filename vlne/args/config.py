"""
Definition of a `Config` class that parametrizes training.
"""

import hashlib
import json
import os

from vlne.consts import DEF_SEED
from .funcs import modify_vars

CONFIG_FNAME = 'config.json'

class Config:
    # pylint: disable=too-many-instance-attributes
    """Training configuration.

    `Config` is a structure that holds parameters required to reproduce the
    training.

    Parameters
    ----------
    batch_size : int
        Training batch size.
    dataset : str
        Dataset path inside "${VLNE_DATADIR}".
    early_stop : dict or None, optional
        Early stopping configuration.
        C.f. `vlne.train.setup.get_early_stop` for available configurations.
        If None, no early stopping will be used. Default: None.
    epochs : int
        Number of epochs training will be run.
    max_prongs : int or None, optional
        Limit number of 3d prongs to `max_prongs`. In other words, if the
        number of 3d prongs is greater than `max_prongs` the remaining prongs
        will be discarded. If None no prong limit will be applied.
        Default: None.
    model : dict
        Network configuration.
        C.f. `vlne.train.setup.select_model` for available configurations.
    noise : dict or list of dict or None, optional
        Noise configuration to be added during the training phase.
        If `noise` is dict then single noise will be added parametrized by
        `noise.  The `noise` parameters will be passed to a constructor of the
        `DataNoise` object defined in `data.data_generator.data_noise`.
        If `noise` is list of dict, then multiple noises will be applied
        sequentially as defined by each dict in the list.
        If `noise` is None, then no noise will be applied.
        Default: None.
    optimizer : dict
        Optimizer configuration to use for training.
        C.f. `vlne.train.setup.get_optimizer` for the available options.
    prong_sorter : dict or None, optional
        Prong sorting specifications to use for 2D and 3D prongs of the form
        { 'input_png2d' : PRONG_SORT_TYPE, 'input_png3d' : PRONG_SORT_TYPE }.
        C.f. `name` argument of the constructor of `DataProngSorter` defined
        in `vlne.data.data_generator.data_prong_sorter` for the list of
        available sorting types. If `prong_sorter` is None then no prong
        sorting will be used. Default: None.
    regularizer : dict or None, optional
        Regularization configuration to be used during the training.
        C.f. `vlne.train.setup.get_regularizer` for the available options.
        If None, no regularization will be used. Default: None.
    schedule : dict or None, optional
        Learning rate decay schedule configuration.
        C.f. `vlne.train.setup.get_schedule` for the available options.
        If None, no schedule will be used. Default: None.
    seed : int
        Seed to be used for the pseudo-random generator initialization.
        This seed affects:
          - data shuffling
          - data train/test split
          - prong sorting in case of randomized prong order
          - noise applied to the data (if any)
          - training itself
    shuffle_data : bool
        Whether to shuffle dataset.
    steps_per_epoch : int or None, optional
        Number of batches to use per training epoch. If None then all available
        batches will be used in a single epoch. Default: None.
    test_size : int or float
        Amount of the `dataset` to be used for network validation
        (aka validation set or dev set).
        If `test_size` is int, then `test_size` entries will be uniformly
        randomly taken from the `dataset` as a validation sample.
        If `test_size` is float and `test_size` < 1, then a fraction
        `test_size` of the `dataset` will be held as validation sample
        (also sampled uniformly at random).
    vars_input_slice : list of str or None, optional
        Names of slice level input variables. If None then no slice level
        inputs will be used. Default: None.
    vars_input_png2d : list of str or None, optional
        Names of 2D prong level input variables. If None then no 2D prong level
        inputs will be used. Default: None.
    vars_input_png3d : list of str or None, optional
        Names of 3D prong level input variables. If None then no 3D prong level
        inputs will be used. Default: None.
    var_target_total : str or None, optional
        Name of the variable that defines true total energy (e.g. neutrino
        energy) in the event.
        If None, no total energy will be predicted by the model.
        Default: None.
    var_primary_total : str or None, optional
        Name of the variable that defined true primary energy (e.g. lepton
        energy) in the event.
        If None, no primary energy will be predicted by the model.
        Default: None.
    weights : dict or None, optional
        Configuration of weights to use.
        C.f. `vlne.data.data.get_weights` for a list of available options.
        If None, all events will be equally weighted. Default: None
    """

    __slots__ = (
        'batch_size',
        'dataset',
        'early_stop',
        'epochs',
        'loss',
        'max_prongs',
        'model',
        'noise',
        'optimizer',
        'prong_sorters',
        'regularizer',
        'schedule',
        'seed',
        'shuffle_data',
        'steps_per_epoch',
        'test_size',
        'vars_input_slice',
        'vars_input_png2d',
        'vars_input_png3d',
        'var_target_total',
        'var_target_primary',
        'weights',
    )

    def __init__(
        self,
        batch_size         = 32,
        dataset            = None,
        early_stop         = None,
        epochs             = 100,
        loss               = None,
        max_prongs         = None,
        model              = None,
        noise              = None,
        optimizer          = None,
        prong_sorters      = None,
        regularizer        = None,
        schedule           = None,
        seed               = DEF_SEED,
        shuffle_data       = True,
        steps_per_epoch    = None,
        test_size          = 0.2,
        vars_input_slice   = None,
        vars_input_png2d   = None,
        vars_input_png3d   = None,
        var_target_total   = None,
        var_target_primary = None,
        weights            = None,
    ):
        self.batch_size         = batch_size
        self.dataset            = dataset
        self.early_stop         = early_stop
        self.epochs             = epochs
        self.loss               = loss
        self.max_prongs         = max_prongs
        self.model              = model
        self.noise              = noise
        self.optimizer          = optimizer
        self.prong_sorters      = prong_sorters
        self.regularizer        = regularizer
        self.schedule           = schedule
        self.seed               = seed
        self.shuffle_data       = shuffle_data
        self.steps_per_epoch    = steps_per_epoch
        self.test_size          = test_size
        self.vars_input_slice   = vars_input_slice
        self.vars_input_png2d   = vars_input_png2d
        self.vars_input_png3d   = vars_input_png3d
        self.var_target_total   = var_target_total
        self.var_target_primary = var_target_primary
        self.weights            = weights

    def to_dict(self):
        return { x : getattr(self, x) for x in self.__slots__ }

    def save(self, savedir):
        with open(os.path.join(savedir, CONFIG_FNAME), 'wt') as f:
            json.dump(self.to_dict(), f, sort_keys = True, indent = 4)

    @staticmethod
    def load(savedir):
        with open(os.path.join(savedir, CONFIG_FNAME), 'rt') as f:
            kwargs = json.load(f)

        return Config(**kwargs)

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys = True)

    def pprint(self):
        return json.dumps(self.to_dict(), sort_keys = True, indent = 4)

    def modify_vars(self, vars_mod_slice, vars_mod_png2d, vars_mod_png3d):
        """Modify input variables.

        This function modifies slice, 2d and 3d prong input variables according
        to the rules defined by the `vars_mod_slice`, `vars_mod_png2d`,
        `vars_mod_png3d` parameters.
        C.f. `Args` constructor for their description.
        """

        self.vars_input_slice = modify_vars(
            self.vars_input_slice, vars_mod_slice
        )

        self.vars_input_png2d = modify_vars(
            self.vars_input_png2d, vars_mod_png2d
        )

        self.vars_input_png3d = modify_vars(
            self.vars_input_png3d, vars_mod_png3d
        )

    def get_hash(self):
        s = json.dumps(self.to_dict(), sort_keys = True)

        md5 = hashlib.md5()
        md5.update(s.encode())

        return md5.hexdigest()

    def get_savedir(self, outdir, label = None):
        if label is None:
            label = self.get_hash()

        savedir = 'model_m(%s)_%s' % (self.model['name'], label)
        savedir = savedir.replace('/', ':')
        path    = os.path.join(outdir, savedir)

        os.makedirs(path, exist_ok = True)
        return path

