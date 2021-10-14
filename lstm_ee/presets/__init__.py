"""
A collection of presets for different flavors of `lstm_ee` trainings/evals.
"""

from .dune   import add_dune_train_presets,   add_dune_eval_presets
from .nova   import add_nova_train_presets,   add_nova_eval_presets
from .uboone import add_uboone_train_presets, add_uboone_eval_presets

PRESETS_TRAIN = {}
PRESETS_EVAL  = {}

add_dune_train_presets(PRESETS_TRAIN)
add_dune_eval_presets(PRESETS_EVAL)

add_nova_train_presets(PRESETS_TRAIN)
add_nova_eval_presets(PRESETS_EVAL)

add_uboone_train_presets(PRESETS_TRAIN)
add_uboone_eval_presets(PRESETS_EVAL)

