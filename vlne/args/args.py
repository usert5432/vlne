"""
Definition of the `Args` object that holds runtime training configuration.
"""

import difflib
import os

from vlne.consts import ROOT_DATADIR, ROOT_OUTDIR
from .config import Config

FNAME_LABEL = 'label'

def get_config_difference(old_conf_str, new_conf_str):
    diff = difflib.unified_diff(
        old_conf_str.split('\n'), new_conf_str.split('\n'),
        fromfile = 'Old Config',
        tofile   = 'New Config',
    )

    return '\n'.join(diff)

class Args:
    # pylint: disable=too-many-instance-attributes
    """Runtime training configuration.

    `Args` contains an instance of `Config` that defines the training, plus a
    number of options that are not necessary for training reproduction.

    Parameters
    ----------
    save_best : bool, optional
        Flag that controls whether save best model according to the validation
        loss, or simply the last model. If True then save the best model,
        otherwise will save the last model. Default: True.
    outdir : str
        Parent directory under `root_outdir` where model directory will be
        created.
    vars_mod_png2d : list of str or None, optional
        A list of strings that define how 2d prong inputs from `Config` should
        be modified. E.g. "+var_name" will add "var_name" to the 2d prong input
        list and  "-var_name" will remove "var_name" from the 2d prong input
        list. If None then `config` 2d prong inputs will not be modified.
        Default: None.
    vars_mod_png3d : list of str or None, optional
        A list of strings that define how 3d prong inputs from `Config` should
        be modified. C.f. `vars_mod_png2d` parameter.
    vars_mod_slice : list of str or None, optional
        A list of strings that define how slice inputs from `Config` should be
        modified. C.f. `vars_mod_png2d` parameter.
    cache : bool, optional
        If True data batches will be cached in RAM. Default: False.
    precache : bool, optional
        If True data batches will be precached in RAM. Default: False.
    workers : int or None, optional
        Number of parallel workers to spawn for the purpose of data batch
        generation. If None then no parallelization will be used.
    **kwargs : dict
        Parameters to be passed to the `Config` constructor.
    extra_kwargs : dict or None, optional
        Specifies extra arguments that will be used to modify `kwargs` above.
        This `extra_kwargs` will be saved in a separate file and will be
        used to determine model `savedir`.

    Attributes
    ----------
    config : Config
        Training configuration.
    savedir : str
        Directory under `root_outdir` where trained model and its config
        will be saved.  It is calculated based on the `outdir` and
        `extra_kwargs` parameters following the pattern:
        `savedir` = `outdir`/model_`extra_kwargs`_hash(hash of `config`).
    root_data : str
        Parent directory where all data is saved.
        Unless set explicitly it is equal to "${LSMT_EE_DATADIR}".
    root_outdir : str
        Parent directory where all trained models are saved.
        Unless set explicitly it is equal to "${LSMT_EE_OUTDIR}".
    """

    # pylint: disable=access-member-before-definition
    __slots__ = (
        'config',
        'savedir',
        'label',
        'save_best',

        'root_datadir',
        'root_outdir',

        'cache',
        'precache',
        'workers',

        'log_level',
    )

    def __init__(
        self, config, savedir,
        label        = None,
        root_datadir = ROOT_DATADIR,
        root_outdir  = ROOT_OUTDIR,
        cache        = False,
        precache     = False,
        save_best    = True,
        workers      = None,
        log_level    = 'INFO',
    ):
        self.config       = config
        self.savedir      = savedir
        self.label        = label
        self.root_datadir = root_datadir
        self.root_outdir  = root_outdir
        self.cache        = (cache or precache)
        self.precache     = precache
        self.save_best    = save_best
        self.workers      = workers
        self.log_level    = log_level

    def _verify_config_collision(self):
        if not os.path.exists(self.savedir):
            return

        try:
            old_config = Config.load(self.savedir)
        except IOError:
            return

        new_conf_str = self.config.pprint()
        old_conf_str = old_config.pprint()

        if old_conf_str != new_conf_str:
            diff = get_config_difference(old_conf_str, new_conf_str)
            raise ValueError(
                f"Config collision detected in '{self.savedir}' :\n{diff}\n"
            )

    def save(self):
        self._verify_config_collision()
        self.config.save(self.savedir)

        if self.label is not None:
            with open(os.path.join(self.savedir, FNAME_LABEL), 'wt') as f:
                f.write(self.label)

    @staticmethod
    def load(savedir):
        config = Config.load(savedir)
        label  = None

        path_label = os.path.join(savedir, FNAME_LABEL)

        if os.path.exists(path_label):
            with open(path_label, 'rt') as f:
                label = f.read()

        return Args(config, savedir, label)

    @staticmethod
    def from_args_dict(
        outdir,
        label          = None,
        root_datadir   = ROOT_DATADIR,
        root_outdir    = ROOT_OUTDIR,
        cache          = False,
        precache       = False,
        save_best      = True,
        workers        = 0,
        log_level      = 'INFO',
        **conf_dict
    ):
        config  = Config(**conf_dict)
        savedir = config.get_savedir(os.path.join(root_outdir, outdir), label)

        result = Args(
            config, savedir, label, root_datadir, root_outdir, cache, precache,
            save_best, workers, log_level
        )

        result.save()
        return result

    def __getattr__(self, name):
        """Get attribute `name` from `Args.config`.

        This function is invoked when one has called `Args.name`, but the
        `Args` itself does not have `name` attribute. In such case it will
        return `Args.config.name`.
        """
        return getattr(self.config, name)

    def __getitem__(self, name):
        """Get `Args` or `Args.config` attribute specified by address `name`.

        This function tries to return an attribute of either `Args` or
        `Args.config` that is encoded in address `name`.

        Parameters
        ----------
        name : str of list of str
            If `name` is str, then first it is converted to a list of str by
            splitting it using ':' as delimiter. Say the resulting list is
            [ "attr", "addr1", "addr2" ]. Then this function will return
            value of the expression: getattr(`self`, "attr")["addr1"]["addr2"].

            If `name` is list of str, then it will return a list of
            [ self[x] for x in `name` ]

        Returns
        -------
        str or list of str
            Value(s) specified by `name`.
        """

        # NOTE: Address is needed to access keys of the nested dictionaries.
        #       e.g. say config.opt_kwargs = { 'lr' : 0.001 }
        #       To access learning rate 'lr' directly one has to call:
        #       args['opt_kwargs:lr']

        if isinstance(name, list):
            return [self[n] for n in name]

        address = name.split(':')

        result = getattr(self, address[0])

        for addr_part in address[1:]:
            result = result[addr_part]

        return result

