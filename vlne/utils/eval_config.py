"""
Definition of `EvalConfig` that holds parameters of evaluation.
"""

import json
import os

EVAL_CONFIG_FNAME = 'evsl_config.json'

def modify_args_value_from_conf(args, attr, eval_value):
    """Load `args` attribute `attr` from a json file `eval_value`"""
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
    transform : str or None,
        JSON file name with the transform config that will be used during
        evaluation.  C.f. `DataConfig.transform` for the configuration spec.
    split : str
        Data split to use for evaluation. Supported: [ 'train', 'val', 'test' ]
    preset : str or None,
        Name of the evaluation preset. C.f. `PRESETS_EVAL`.
    weights : str or None
        Weights specification that will be used during the evaluation.
        C.f. `DataConfig.weights`.
    """

    # pylint: disable=too-many-instance-attributes
    __slots__ = [
        'label',
        'data',
        'preset',
        'split',
        'transform',
        'weights',
    ]

    @staticmethod
    def from_cmdargs(cmdargs):
        """Construct `EvalConfig` from parameters from `argparse.Namespace`"""
        return EvalConfig(
            cmdargs.label,
            cmdargs.data,
            cmdargs.preset,
            cmdargs.split,
            cmdargs.transform,
            cmdargs.weights,
        )

    def __init__(
        self, label, data, preset, split, transform, weights
    ):
        self.label     = label
        self.data      = data
        self.preset    = preset
        self.split     = split
        self.transform = transform
        self.weights   = weights

    def to_dict(self):
        return { k : getattr(self, k) for k in self.__slots__ }

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)

    def get_evaldir(self, outdir):
        if self.data not in  [ None, 'same' ]:
            data = os.path.basename(self.data)
        else:
            data = 'same'

        return os.path.join(
            outdir,
            f'eval_d({data})_p({self.preset})_s({self.split})_{self.label}'
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

    def modify_data_weights(self, args):
        if (self.weights is None) or (self.weights == 'same'):
            return

        if self.weights == 'none':
            args.config.data.weights = None
            return

        for w in args.config.data.weights:
            args.config.data.weights[w] = self.weights

    def modify_data_path(self, args):
        if (self.data is None) or (self.data == 'same'):
            return

        args.config.data.frame['path'] = self.data
        args.config.data.val_size      = 0
        args.config.data.test_size     = 1.0

    def modify_eval_args(self, args):
        self.modify_data_path(args)
        self.modify_data_weights(args)

        modify_args_value_from_conf(
            args.config.data, 'transform_test', self.transform
        )

