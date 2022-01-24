"""Presets for the NuE energy estimator training/eval"""

from vlne.consts import LABEL_PRIMARY, LABEL_TOTAL
from vlne.presets.eval_preset import EvalPreset

BASE_MAP_DUNE = {
    LABEL_PRIMARY : "numue.lepE",
    LABEL_TOTAL   : "numue.nuE",
}

preset_dune_numu_base = {
    'var_target_total'   : 'mc.nuE',
    'var_target_primary' : 'mc.lepE',
}

preset_dune_numu_v1 = {
    **preset_dune_numu_base,
    'vars_input_slice': [
        "event.calE",
        "event.charge",
        "event.nHits",
    ],
    'vars_input_png3d' : [
        "particle.is_shower",
        "particle.length",
        "particle.start.x",
        "particle.start.y",
        "particle.start.z",
        "particle.dir.x",
        "particle.dir.y",
        "particle.dir.z",
        "particle.energy",
        "particle.calE",
        "particle.charge",
        "particle.nHit",
    ],
}

def add_dune_train_presets(presets):
    presets['dune_numu_v1'] = preset_dune_numu_v1

def add_dune_eval_presets(presets):
    presets['dune_numu'] = EvalPreset(
        name_overrides = {
            LABEL_PRIMARY : 'Muon',
        },
        base_overrides = BASE_MAP_DUNE,
    )

