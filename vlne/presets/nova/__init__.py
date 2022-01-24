"""Presets for the NuMu energy estimator"""

from .numu import add_train_numu_presets, add_eval_numu_presets
from .nue  import add_train_nue_presets,  add_eval_nue_presets

def add_nova_train_presets(presets):
    add_train_numu_presets(presets)
    add_train_nue_presets(presets)

def add_nova_eval_presets(presets):
    add_eval_numu_presets(presets)
    add_eval_nue_presets(presets)

