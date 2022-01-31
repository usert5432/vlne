"""
A collection of presets for different flavors of `lstm_ee` trainings/evals.
"""

from .dune.presets import (
    add_train_dune_numu_presets,add_eval_dune_numu_presets
)
from .uboone.presets import (
    add_train_uboone_numu_presets,add_eval_uboone_numu_presets
)
from .numu.presets import add_train_numu_presets,add_eval_numu_presets
from .nue.presets  import add_train_nue_presets,add_eval_nue_presets

PRESETS_TRAIN = {}
PRESETS_EVAL  = {}

add_train_numu_presets(PRESETS_TRAIN)
add_eval_numu_presets(PRESETS_EVAL)

add_train_nue_presets(PRESETS_TRAIN)
add_eval_nue_presets(PRESETS_EVAL)

add_train_dune_numu_presets(PRESETS_TRAIN)
add_eval_dune_numu_presets(PRESETS_EVAL)

add_train_uboone_numu_presets(PRESETS_TRAIN)
add_eval_uboone_numu_presets(PRESETS_EVAL)

