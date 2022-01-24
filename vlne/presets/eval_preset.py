import copy
from vlne.consts import LABEL_TOTAL, LABEL_PRIMARY, LABEL_SECONDARY

DEF_UNITS_MAP = {
    LABEL_PRIMARY   : 'GeV',
    LABEL_SECONDARY : 'GeV',
    LABEL_TOTAL     : 'GeV',
}

DEF_NAME_MAP = {
    LABEL_PRIMARY   : 'Lepton',
    LABEL_SECONDARY : 'Hadronic',
    LABEL_TOTAL     : 'Neutrino',
}

DEF_BASE_MAP = {
    LABEL_PRIMARY : "reco.lepE",
    LABEL_TOTAL   : "reco.nuE",
}

class EvalPreset:

    __slots__ = [
        'name_map',
        'units_map',
        'base_map',
    ]

    @staticmethod
    def _init_map(default, overrides):
        result = copy.copy(default)

        if overrides is not None:
            for (k,v) in overrides.items():
                result[k] = v

        return result

    def __init__(
        self,
        name_overrides  = None,
        units_overrides = None,
        base_overrides  = None
    ):
        self.name_map  = EvalPreset._init_map(DEF_NAME_MAP,  name_overrides)
        self.units_map = EvalPreset._init_map(DEF_UNITS_MAP, units_overrides)
        self.base_map  = EvalPreset._init_map(DEF_BASE_MAP,  base_overrides)

