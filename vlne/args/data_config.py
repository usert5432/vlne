import logging
import re

from vlne.consts import DEF_SEED, LABEL_TOTAL, LABEL_PRIMARY
from .config_base import ConfigBase
from .funcs import modify_vars

LOGGER = logging.getLogger('vlne.data')

DEPRECATED_WEIGHT = 'vlne.weight.deprecated'

class DataConfig(ConfigBase):

    # pylint: disable=too-many-instance-attributes
    __slots__ = (
        'frame',
        'extra_vars',
        'input_groups_scalar',
        'input_groups_vlarr',
        'target_groups',
        'vlarr_limits',
        'transform_train',
        'transform_test',
        'val_size',
        'test_size',
        'seed',
        'shuffle',
        'weights',
    )

    def __init__(
        self,
        frame               = None,
        extra_vars          = None,
        input_groups_scalar = None,
        input_groups_vlarr  = None,
        target_groups       = None,
        vlarr_limits        = None,
        transform_train     = None,
        transform_test      = None,
        val_size            = None,
        test_size           = None,
        seed                = 0,
        shuffle             = None,
        weights             = None,
    ):
        self.frame               = frame
        self.extra_vars          = extra_vars
        self.input_groups_scalar = input_groups_scalar
        self.input_groups_vlarr  = input_groups_vlarr
        self.target_groups       = target_groups
        self.vlarr_limits        = vlarr_limits
        self.transform_train     = transform_train
        self.transform_test      = transform_test
        self.val_size            = val_size
        self.test_size           = test_size
        self.seed                = seed
        self.shuffle             = shuffle
        self.weights             = weights

def parse_prong_sorter_transform(prong_sorters):
    if prong_sorters is None:
        return []

    result = []
    regexp = re.compile(r'^([-\+])(.*)$')

    for (group, sorter) in prong_sorters.items():
        if sorter == 'random':
            result.append({ 'name' : 'vlarr-shuffle', 'vlarr_group' : group })

        elif sorter is None:
            continue

        else:
            m = regexp.match(sorter)

            result.append({
                'name'        : 'vlarr-sort',
                'vlarr_group' : group,
                'column'      : m.group(2),
                'ascending'   : (m.group(1) == '+'),
            })

    return result

def parse_single_noise(noise):
    result = {
        'name'       : 'noise',
        'noise'      : {
            'name' : noise['noise'],
            **noise['noise_kwargs'],
        },
        'correlated' : True,
        'relative'   : True,
    }

    scalar_groups = { }
    vlarr_groups  = { }

    slice_vars = noise.get('affected_vars_slice', None)
    png3d_vars = noise.get('affected_vars_png3d', None)
    png2d_vars = noise.get('affected_vars_png2d', None)

    if slice_vars is not None:
        scalar_groups['input_slice'] = slice_vars

    if png2d_vars is not None:
        vlarr_groups['input_png2d'] = png2d_vars

    if png3d_vars is not None:
        vlarr_groups['input_png3d'] = png3d_vars

    result['scalar_groups'] = scalar_groups or None
    result['vlarr_groups']  = vlarr_groups  or None

    return result

def parse_deprecated_var_groups(
    vars_input_slice   = None,
    vars_input_png2d   = None,
    vars_input_png3d   = None,
    var_target_total   = None,
    var_target_primary = None,
    vars_mod_slice     = None,
    vars_mod_png2d     = None,
    vars_mod_png3d     = None,
    max_prongs         = None,
):
    input_groups_scalar = {}
    input_groups_vlarr  = {}
    target_groups = {}
    vlarr_limits  = None

    if vars_input_slice is not None:
        vars_input_slice = modify_vars(vars_input_slice, vars_mod_slice)
        input_groups_scalar['input_slice'] = vars_input_slice

    if var_target_total is not None:
        target_groups[LABEL_TOTAL] = [ var_target_total, ]

    if var_target_primary is not None:
        target_groups[LABEL_PRIMARY] = [ var_target_primary, ]

    if vars_input_png3d is not None:
        vars_input_png3d = modify_vars(vars_input_png3d, vars_mod_png3d)
        input_groups_vlarr['input_png3d'] = vars_input_png3d

    if vars_input_png2d is not None:
        vars_input_png2d = modify_vars(vars_input_png2d, vars_mod_png2d)
        input_groups_vlarr['input_png2d'] = vars_input_png2d

    if max_prongs is not None:
        vlarr_limits = { k : max_prongs for k in input_groups_vlarr }

    return input_groups_scalar, input_groups_vlarr, target_groups, vlarr_limits

def guess_frame_name(path):
    if isinstance(path, dict):
        return 'dict-frame'

    if isinstance(path, str):
        for ext in [ 'h5', 'hdf', 'hdf5' ]:
            if path.endswith(ext):
                return 'hdf-ra-frame'

        return 'csv-mem-frame'

    raise RuntimeError("Unknown how to load data: %s" % (path))

def parse_deprecated_frame(dataset):
    return {
       'name' : guess_frame_name(dataset),
       'path' : dataset
    }

def parse_deprecated_transforms(noise, prong_sorters):
    result = [ 'mask-nan' ]

    if prong_sorters is not None:
        result += parse_prong_sorter_transform(prong_sorters)

    if noise is not None:
        if isinstance(noise, dict):
            noise = [ noise, ]

        result += [ parse_single_noise(n) for n in noise ]

    return result

def parse_deprecated_weights(
    var_target_total   = None,
    var_target_primary = None,
    weights_spec       = None,
):
    if weights_spec is None:
        return None, None

    weights = { }

    if var_target_total is not None:
        weights[LABEL_TOTAL] = DEPRECATED_WEIGHT

    if var_target_primary is not None:
        weights[LABEL_PRIMARY] = DEPRECATED_WEIGHT

    extra_vars = { DEPRECATED_WEIGHT : weights_spec }

    return (extra_vars, weights)

def parse_deprecated_config(
    dataset            = None,
    max_prongs         = None,
    noise              = None,
    prong_sorters      = None,
    seed               = DEF_SEED,
    shuffle_data       = True,
    test_size          = 0.2,
    vars_input_slice   = None,
    vars_input_png2d   = None,
    vars_input_png3d   = None,
    var_target_total   = None,
    var_target_primary = None,
    vars_mod_slice     = None,
    vars_mod_png2d     = None,
    vars_mod_png3d     = None,
    weights            = None,
):
    input_groups_scalar, input_groups_vlarr, target_groups, vlarr_limits \
        = parse_deprecated_var_groups(
            vars_input_slice, vars_input_png2d, vars_input_png3d,
            var_target_total, var_target_primary,
            vars_mod_slice, vars_mod_png2d, vars_mod_png3d,
            max_prongs
        )

    transforms = parse_deprecated_transforms(noise, prong_sorters)
    extra_vars, weights = parse_deprecated_weights(
        var_target_total, var_target_primary, weights
    )

    return DataConfig(
        frame               = parse_deprecated_frame(dataset),
        extra_vars          = extra_vars,
        input_groups_scalar = input_groups_scalar,
        input_groups_vlarr  = input_groups_vlarr,
        target_groups       = target_groups,
        vlarr_limits        = vlarr_limits,
        transform_train     = transforms,
        transform_test      = transforms,
        val_size            = test_size,        # Not a mistake
        test_size           = None,             # This is for bkw. compat.
        seed                = seed,
        shuffle             = shuffle_data,
        weights             = weights,
    )

def parse_data_config(
    data = None,
    seed = None,
    # Deprecated options:
    dataset            = None,
    max_prongs         = None,
    noise              = None,
    prong_sorters      = None,
    shuffle_data       = None,
    test_size          = None,
    vars_input_slice   = None,
    vars_input_png2d   = None,
    vars_input_png3d   = None,
    var_target_total   = None,
    var_target_primary = None,
    vars_mod_slice     = None,
    vars_mod_png2d     = None,
    vars_mod_png3d     = None,
    weights            = None,
):
    # pylint: disable=too-many-boolean-expressions
    if (
           (dataset            is not None)
        or (max_prongs         is not None)
        or (noise              is not None)
        or (prong_sorters      is not None)
        or (shuffle_data       is not None)
        or (test_size          is not None)
        or (vars_input_slice   is not None)
        or (vars_input_png2d   is not None)
        or (vars_input_png3d   is not None)
        or (var_target_total   is not None)
        or (var_target_primary is not None)
        or (vars_mod_slice     is not None)
        or (vars_mod_png2d     is not None)
        or (vars_mod_png3d     is not None)
        or (weights            is not None)
   ):
        if (data is not None):
            raise ValueError("Cannot mix old and new data configuration")

        LOGGER.warning("Deprecated data configuration detected")

        if seed is None:
            seed = DEF_SEED

        return parse_deprecated_config(
            dataset,  max_prongs, noise, prong_sorters, seed, shuffle_data,
            test_size,
            vars_input_slice, vars_input_png2d, vars_input_png3d,
            var_target_total, var_target_primary,
            vars_mod_slice, vars_mod_png2d, vars_mod_png3d,
            weights
        )

    return DataConfig(**data)

