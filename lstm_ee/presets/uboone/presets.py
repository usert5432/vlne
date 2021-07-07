"""Presets for the NuE energy estimator training/eval"""

from lstm_ee.presets.numu.specs import (
    get_numu_eval_presets, LABEL_PRIMARY, LABEL_TOTAL
)

BASE_MAP_DUNE = {
    LABEL_PRIMARY : "reco.lepE",
    LABEL_TOTAL   : "reco.nuE",
}

preset_uboone_numu_base = {
    'var_target_total'   : 'truth.nuE',
    'var_target_primary' : 'truth.lepE',
}

preset_uboone_numu_v1 = {
    **preset_uboone_numu_base,
    'vars_input_slice': [
        "event.numu_score",
    ],
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

def get_uboone_numu_eval_presets(energy_range):
    """Construct MicroBooNE numu eval presets from NOvA preset"""
    result = get_numu_eval_presets(energy_range)
    result['base_map'] = BASE_MAP_DUNE

    return result

def add_train_uboone_numu_presets(presets):
    """Add MicroBooNE numu training presets to the `presets` dict"""
    presets['uboone_numu_v1'] = preset_uboone_numu_v1

def add_eval_uboone_numu_presets(presets):
    """Add MicroBooNE numu eval presets to the `presets` dict"""
    presets['uboone_numu_5GeV'] = get_uboone_numu_eval_presets((0, 6))

