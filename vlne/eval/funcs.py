import numpy as np
from vlne.consts import LABEL_TOTAL

def get_weights(weights_dict, target, pred_dict):
    if target in weights_dict:
        return weights_dict[target]

    if LABEL_TOTAL in weights_dict:
        return weights_dict[LABEL_TOTAL]

    return np.ones(len(pred_dict[target]))


