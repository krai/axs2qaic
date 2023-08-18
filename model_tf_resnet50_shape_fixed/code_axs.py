import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
import copy
import os

def load_frozen_graph(model_file):
	graph_def = tf.compat.v1.GraphDef()
	with gfile.FastGFile(model_file, 'rb') as f:
		graph_def.ParseFromString(f.read())
	return graph_def

def modified_model(input_graph, input_name, input_type, model_name, file_name, tags=None, entry_name=None, __record_entry__=None ):
    __record_entry__["tags"] = tags or [ "tf_model", "shape_fixed" ]
    if not entry_name:
        entry_name = f'{model_name}_tf_model_shape_fixed'
    __record_entry__.save( entry_name )
    output_directory        = __record_entry__.get_path( "" )
    output_modified_graph   = __record_entry__.get_path(file_name)
    
    inputGraph = load_frozen_graph(input_graph)
    outputGraph = tf.compat.v1.GraphDef() 
    new_input = tf.placeholder( dtype=tf.float32, shape = [1, 224, 224, 3], name=input_name)
    for node in inputGraph.node:
        if node.name == input_name:
            print("replacing ()".format(input_name))
            outputGraph.node.extend([new_input.op.node_def])
        else:
            outputGraph.node.extend([copy.deepcopy(node)])
    # Save the new graph.
    with tf.compat.v1.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(outputGraph, name="")
        if(input_type == 'b'):
            tf.io.write_graph(sess.graph.as_graph_def(add_shapes=True), os.path.dirname(output_modified_graph), os.path.basename(output_modified_graph), as_text=False)
        if(input_type == 't'):
            tf.io.write_graph(sess.graph.as_graph_def(add_shapes=True), os.path.dirname(output_modified_graph), os.path.basename(output_modified_graph), as_text=True)
    print("The input shape has been fixed successfully.")

    # Cut the unwanted nodes.
    graph = tf.compat.v1.GraphDef()
    with tf.gfile.Open(output_modified_graph, 'rb') as f:
        data = f.read()
        graph.ParseFromString(data)

    nodes = graph.node[2:]
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes)
    with tf.gfile.GFile(output_modified_graph, 'w') as f:
        os.remove(output_modified_graph)
        f.write(output_graph.SerializeToString())

    return __record_entry__
