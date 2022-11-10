"""
Functions to calculate true and predicted energies.
"""
import logging

from vlne.consts import LABEL_TOTAL, LABEL_PRIMARY, LABEL_SECONDARY
from vlne.train.setup import get_keras_concurrency_kwargs

LOGGER = logging.getLogger('vlne.eval')

def fill_missing_energies(result):
    n_nan = sum(1 for x in result.values() if x is None)
    if n_nan != 1:
        return

    if result[LABEL_TOTAL] is None:
        result[LABEL_TOTAL] = result[LABEL_PRIMARY] + result[LABEL_SECONDARY]

    elif result[LABEL_PRIMARY] is None:
        result[LABEL_PRIMARY] = result[LABEL_TOTAL] - result[LABEL_SECONDARY]

    elif result[LABEL_SECONDARY] is None:
        result[LABEL_SECONDARY] = result[LABEL_TOTAL] - result[LABEL_PRIMARY]

    else:
        raise RuntimeError('Unknown labels in energy result dict')

def get_output_by_label(pred, model, label):
    if all(x.startswith('target_') for x in model.output_names):
        LOGGER.warning('Old model format detected. Using workaround...')
        label = 'target_' + label

    if label not in model.output_names:
        return None

    index = model.output_names.index(label)
    if pred[index].shape[1] != 1:
        return None

    return pred[index].ravel()

def predict_energies(args, dgen, model):
    kwargs = get_keras_concurrency_kwargs(args)
    pred   = model.predict(dgen, **kwargs)

    result = {
        LABEL_TOTAL     : get_output_by_label(pred, model, LABEL_TOTAL),
        LABEL_PRIMARY   : get_output_by_label(pred, model, LABEL_PRIMARY),
        LABEL_SECONDARY : get_output_by_label(pred, model, LABEL_SECONDARY),
    }

    fill_missing_energies(result)
    return result

def get_true_energies(dgen):
    result = {
        LABEL_TOTAL     : None,
        LABEL_PRIMARY   : None,
        LABEL_SECONDARY : None,
    }

    for label in [ LABEL_TOTAL, LABEL_PRIMARY, LABEL_SECONDARY ]:
        if (
                (label in dgen.target_groups)
            and (len(dgen.dataset.scalar_groups[label]) == 1)
        ):
            variable      = dgen.dataset.scalar_groups[label][0]
            result[label] = dgen.dataset.df[variable].ravel()

    fill_missing_energies(result)
    return result

def get_base_energies(dgen, pred_map):
    if pred_map is None:
        return None

    result = {
        LABEL_TOTAL     : None,
        LABEL_PRIMARY   : None,
        LABEL_SECONDARY : None,
    }

    for label in [ LABEL_TOTAL, LABEL_PRIMARY, LABEL_SECONDARY ]:
        if label in pred_map:
            result[label] = dgen.dataset.df[pred_map[label]].ravel()

    fill_missing_energies(result)
    return result

