"""Presets for the NuE energy estimator training/eval"""

from vlne.consts import LABEL_PRIMARY, LABEL_TOTAL
from vlne.presets.eval_preset import EvalPreset

BASE_MAP_UBOONE = {
    LABEL_PRIMARY : "reco.lepE",
    LABEL_TOTAL   : "reco.nuE",
}

preset_uboone_numu_base = {
    'var_target_total'   : 'truth.nuE',
    'var_target_primary' : 'truth.lepE',
}

preset_uboone_numu_v1 = {
    **preset_uboone_numu_base,
    'vars_input_slice': [ ],
    'vars_input_png3d' : [
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
        "particle.startMomentum.z",
    ],
}

def add_uboone_train_presets(presets):
    presets['uboone_numu_v1'] = preset_uboone_numu_v1
    presets['uboone_v1'] = preset_uboone_numu_v1

def add_uboone_eval_presets(presets):
    presets['uboone_numu'] = EvalPreset(
        name_overrides = {
            LABEL_PRIMARY : 'Muon',
        },
        base_overrides = BASE_MAP_UBOONE,
    )

    presets['uboone_nue'] = EvalPreset(
        name_overrides = {
            LABEL_PRIMARY : 'Electron',
        },
        base_overrides = BASE_MAP_UBOONE,
    )

    presets['uboone_generic'] = EvalPreset(base_overrides = BASE_MAP_UBOONE)

