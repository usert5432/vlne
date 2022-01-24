"""
Definition of `EvalConfig` that holds parameters of evaluation.
"""

import json
import os

EVAL_CONFIG_FNAME = 'evsl_config.json'

def modify_args_value(args, attr, eval_value, dtype = None):
    """Set `args` attribute `attr` to `eval_value`.

    This function modifies `args` attribute `attr`. It sets it to `eval_value`
    paying attention to the special values of `eval_value`:
      - if `eval_value` is None, then the `attr` value won't be modified.
      - if `eval_value` == 'none', then the `attr` value will be set to None.
      - otherwise, `attr` = `eval_value`
    """
    if (eval_value is None) or (eval_value == 'same'):
        return

    if eval_value.lower() == 'none':
        setattr(args, attr, None)
    else:
        if dtype:
            setattr(args, attr, dtype(eval_value))
        else:
            setattr(args, attr, eval_value)

def modify_args_value_from_conf(args, attr, eval_value):
    """Load `args` attribute `attr` from a json file `eval_value`.

    This function modifies `args` attribute `attr`. It loads its value from
    a json file `eval_value` paying attention to the special values of
    `eval_value`:
      - if `eval_value` is None, then the `attr` value won't be modified.
      - if `eval_value` == 'none', then the `attr` value will be set to None.
      - otherwise, `attr` = json.load(`eval_value`)
    """
    if (eval_value is None) or (eval_value == 'same'):
        return

    if eval_value.lower() == 'none':
        setattr(args, attr, None)
    else:
        with open(eval_value, 'rt') as f:
            setattr(args, attr, json.load(f))

class EvalConfig:
    """Configuration of an `vlne` network evaluation.

    Parameters of the `EvalConfig` will be used to modify the corresponding
    parameters of the `Args` of the trained network. The rules of such
    modification are as follows:
        - If (parameter == "same") or (parameter is None) then the value of
          `Args` will not be modified.
        - If parameter == "none" then the value of `Args` will be set to None.
        - otherwise `Args` parameter will be either directly set to parameter
          from `EvalConfig` or will be loaded from a file specified by a
          parameter of `EvalConfig`.

    Parameters
    ----------
    label : str or None,
        Label used to uniquely identify current evaluation.
    data : str or None,
        Name of the evaluation dataset.
    noise : str or None,
        JSON file name with the noise config that will be used during
        evaluation.  C.f. `Config.noise` for the configuration spec.
    preset : str or None,
        Name of the evaluation preset. C.f. `PRESETS_EVAL`.
    prong_sorter : str or None,
        JSON file name with the prong sorting config that will be used during
        evaluation.  C.f. `Config.prong_sorter` for the configuration spec.
    seed : int or str or None,
        Seed to initialize PRG for splitting dataset into training/validation
        parts.
    test_size : int or float or str or None
        Size of the dataset that will be used for evaluation.
        C.f. `Config.test_size`.
    weights : str or None
        Weights specification that will be used during the evaluation.
        C.f. `Config.weights`.
    """

    # pylint: disable=too-many-instance-attributes
    __slots__ = [
        'label',
        'data',
        'noise',
        'preset',
        'prong_sorter',
        'seed',
        'test_size',
        'weights',
    ]

    @staticmethod
    def from_cmdargs(cmdargs):
        """Construct `EvalConfig` from parameters from `argparse.Namespace`"""
        return EvalConfig(
            cmdargs.label,
            cmdargs.data,
            cmdargs.noise,
            cmdargs.preset,
            cmdargs.prong_sorter,
            cmdargs.seed,
            cmdargs.test_size,
            cmdargs.weights,
        )

    def __init__(
        self, label, data, noise, preset, prong_sorter, seed, test_size,
        weights
    ):
        self.label        = label
        self.data         = data
        self.noise        = noise
        self.preset       = preset
        self.prong_sorter = prong_sorter
        self.seed         = seed
        self.test_size    = test_size
        self.weights      = weights

    def to_dict(self):
        return { k : getattr(self, k) for k in self.__slots__ }

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)

    def get_evaldir(self, outdir):
        return os.path.join(
            outdir,
            'eval_d(%s)_p(%s)_%s' % (
                self.data or 'same', self.preset, self.label
            )
        )

    @staticmethod
    def load(evaldir):
        path = os.path.join(evaldir, EVAL_CONFIG_FNAME)

        with open(path, 'rt') as f:
            return EvalConfig(**json.load(f))

    def save(self, evaldir):
        path = os.path.join(evaldir, EVAL_CONFIG_FNAME)

        curr_conf_str = self.to_json(sort_keys = True, indent = 4)

        if os.path.exists(path):
            with open(path, 'rt') as f:
                prev_conf_str = f.read()

            assert curr_conf_str == prev_conf_str, \
                "Evaluation config collision detected: %s != %s" % (
                    curr_conf_str, prev_conf_str
                )

        with open(path, 'wt') as f:
            f.write(curr_conf_str)

    def modify_eval_args(self, args):
        """Modify parameters of `args` using values from `self`"""
        modify_args_value(args.config, 'dataset',   self.data)
        modify_args_value(args.config, 'seed',      self.seed, int)
        modify_args_value(args.config, 'test_size', self.test_size, float)
        modify_args_value(args.config, 'weights',   self.weights)

        modify_args_value_from_conf(args.config, 'noise', self.noise)
        modify_args_value_from_conf(
            args.config, 'prong_sorters', self.prong_sorter
        )

