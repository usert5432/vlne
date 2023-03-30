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
        model, config_tf, args.config.data.input_groups_scalar,
        'input_groups_scalar'
    )

    save_input_groups_to_config(
        model, config_tf, args.config.data.input_groups_vlarr,
        'input_groups_vlarr'
    )

    config_tf['target_groups'] = {}

    for (key, values) in args.config.data.target_groups.items():
        name = get_tf_opname_for_layer(model, key, True)
        config_tf['target_groups'][name] = len(values)

    return config_tf

def save_config(config_tf, outdir_tf):
    with open(os.path.join(outdir_tf, "config.json"), "wt") as f:
        json.dump(config_tf, f, indent = 4, sort_keys = True)

def export(config, graph, outdir, text_also):
    outdir = os.path.join(outdir, "tf")
    os.makedirs(outdir, exist_ok = True)

    save_config(config, outdir)
    save_graph(graph, outdir, "model.pb", text_also)

def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(cmdargs.outdir, compile = False)
    config_tf   = create_tf_config(args, model)

    inputs  = [ node.op.name for node in model.inputs ]
    outputs = [ node.op.name for node in model.outputs ]

    graph = freeze_session(K.get_session(), outputs)

    if cmdargs.optimize:
        graph = get_optimized_graph(graph, inputs, outputs)

    export(config_tf, graph, cmdargs.outdir, cmdargs.text_also)

if __name__ == '__main__':
    main()

