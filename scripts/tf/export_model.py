"""Extract tensorflow graph from a `keras` model and export it to a file"""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K

from tensorflow.python.framework import dtypes, graph_io
from tensorflow.python.framework import graph_util
from tensorflow.python.tools     import optimize_for_inference_lib

from vlne.utils.io import load_model

tf.compat.v1.disable_eager_execution()
K.set_learning_phase(0)

def parse_cmdargs():
    parser = argparse.ArgumentParser("Convert keras model to TF graph")

    parser.add_argument(
        'outdir',
        help    = 'directory with saved models',
        metavar = 'OUTDIR',
        type    = str,
    )

    parser.add_argument(
        '-t', '--text',
        action = 'store_true',
        dest   = 'text_also',
        help   = 'save ascii version also',
    )

    parser.add_argument(
        '--optimize',
        action = 'store_true',
        dest   = 'optimize',
        help   = 'optimize graph before saving',
    )

    parser.add_argument(
        '--compat',
        action = 'store_true',
        dest   = 'compat',
        help   = 'create config compatible with old vlneval versions',
    )

    return parser.parse_args()

def freeze_session(session, output_names):
    graph = session.graph

    with graph.as_default():
        input_graph_def = graph.as_graph_def()

        for node in input_graph_def.node:
            node.device = ""

        frozen_graph = graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names
        )

        return frozen_graph

def get_optimized_graph(frozen_graph, inputs, outputs):
    DEF_DTYPE = dtypes.float32.as_datatype_enum
    result    = frozen_graph

    print(
        "[WARNING] Running graph optimization. "
        "Note, that TensorFlow likes to break when using optimized graph."
    )

    result = optimize_for_inference_lib.optimize_for_inference(
        result, inputs, outputs, DEF_DTYPE
    )

    return result

def save_graph(graph, root, name, text_also = False):
    graph_io.write_graph(graph, root, name, as_text = False)

    if text_also:
        graph_io.write_graph(graph, root, name + ".txt", as_text = True)

def get_tf_opname_for_layer(model, layer_name, output_op = False):
    """Return tf i/o node name for a layer of keras model"""
    try:
        layer = model.get_layer(layer_name)
    except ValueError:
        return None

    if output_op:
        return layer.output.op.name
    else:
        return layer.input.op.name

def save_input_groups_to_config(model, config_tf, input_groups, label):
    config_tf[label] = {}

    for (key, values) in input_groups.items():
        node = get_tf_opname_for_layer(model, key, False)

        if node is not None:
            config_tf[label][node] = values

def create_tf_config(args, model):
    config_tf = {}

    save_input_groups_to_config(
        model, config_tf, args.config.data.input_groups_scalar, 'inputs_scalar'
    )

    save_input_groups_to_config(
        model, config_tf, args.config.data.input_groups_vlarr, 'inputs_vlarr'
    )

    config_tf['targets'] = {}

    for (key, values) in args.config.data.target_groups.items():
        node = get_tf_opname_for_layer(model, key, True)
        config_tf['targets'][key] = (node, len(values))

    return config_tf

def save_vars_compat(config_tf, label, var_list):
    if var_list is None:
        return

    config_tf[label] = var_list

def save_scalar_vars_compat(config_tf, scalar_groups, compat_map):
    assert len(scalar_groups) == 1,\
        "Compat mode support single scalar input group only"

    group_name = next(iter(scalar_groups))

    save_vars_compat(config_tf, 'vars_event', scalar_groups[group_name])
    compat_map['input_event'] = group_name

def save_vlarr_vars_compat(config_tf, vlarr_groups, compat_map):
    assert (len(vlarr_groups) >= 1) and (len(vlarr_groups) <= 2),\
        "Compat mode supports 1 or 2 vlarr input groups only"

    COMPAT_PARTICLE_NODES     = set([
        'input_pnt3d', 'input_particle', 'input_vlarr'
    ])
    COMPAT_PARTICLE_AUX_NODES = set([ 'input_pnt2d', 'input_particle_aux' ])

    for (group_name, values) in vlarr_groups.items():
        if group_name in COMPAT_PARTICLE_NODES:
            save_vars_compat(config_tf, 'vars_particle', values)
            compat_map['input_particle'] = group_name

        elif group_name in COMPAT_PARTICLE_AUX_NODES:
            save_vars_compat(config_tf, 'vars_particle_aux', values)
            compat_map['input_particle_aux'] = group_name

        else:
            raise ValueError(
                "Cannot find compatible interpretation of variable group"
                f" {group_name}"
            )

def find_target_vars_compat(target_groups):
    assert (len(target_groups) >= 1) and (len(target_groups) <= 2),\
        "Compat mode supports 1 or 2 target groups only"

    target_compat_map = {}

    COMPAT_TOTAL_NODES   = set([ 'total',   'target_total' ])
    COMPAT_PRIMARY_NODES = set([ 'primary', 'target_primary' ])

    for group_name in target_groups:
        if group_name in COMPAT_TOTAL_NODES:
            target_compat_map['target_total'] = group_name

        elif group_name in COMPAT_PRIMARY_NODES:
            target_compat_map['target_primary'] = group_name

        else:
            print(
                "[NOTE] Cannot find compatible interpretation of target group"
                f" {group_name}"
            )

    return target_compat_map

def create_tf_config_compat(args, model):
    config_tf  = {}
    compat_map = {}

    scalar_groups = args.config.data.input_groups_scalar
    vlarr_groups  = args.config.data.input_groups_vlarr

    save_scalar_vars_compat(config_tf, scalar_groups, compat_map)
    save_vlarr_vars_compat (config_tf, vlarr_groups,  compat_map)

    for (key, name) in compat_map.items():
        node = get_tf_opname_for_layer(model, name, False)
        if node is not None:
            config_tf[key] = node

    target_compat = find_target_vars_compat(args.config.data.target_groups)

    config_tf.update({
        key : get_tf_opname_for_layer(model, name, True) \
            for (key, name) in target_compat.items()
    })

    return config_tf

def save_config(config_tf, outdir_tf):
    with open(os.path.join(outdir_tf, "config.json"), "wt") as f:
        json.dump(config_tf, f, indent = 4, sort_keys = True)

def export(config, graph, outdir, text_also, compat):
    if compat:
        outdir = os.path.join(outdir, "tf_compat")
    else:
        outdir = os.path.join(outdir, "tf")

    os.makedirs(outdir, exist_ok = True)

    save_config(config, outdir)
    save_graph(graph, outdir, "model.pb", text_also)

def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(cmdargs.outdir, compile = False)

    if cmdargs.compat:
        config_tf = create_tf_config_compat(args, model)
    else:
        config_tf = create_tf_config(args, model)

    inputs  = [ node.op.name for node in model.inputs ]
    outputs = [ node.op.name for node in model.outputs ]

    graph = freeze_session(K.get_session(), outputs)

    if cmdargs.optimize:
        graph = get_optimized_graph(graph, inputs, outputs)

    export(config_tf, graph, cmdargs.outdir, cmdargs.text_also, cmdargs.compat)

if __name__ == '__main__':
    main()

