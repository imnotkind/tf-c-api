import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


# Freeze the graph

input_graph_path = 'model/fcn.pb'
checkpoint_path = 'model/fcn-ckpt/model.ckpt-haebin' #prefix of checkpoint, only need .index and .data-???, not .meta
input_saver_def_path = ""
input_binary = True
output_node_names = "Pred"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'model/frozen_fcn.pb'
output_optimized_graph_name = 'model/frozen_fcn_opt.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



# Optimize for inference

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["input_image","keep_probability"], # an array of the input node(s)
        ["Pred"], # an array of output nodes
        tf.float32.as_datatype_enum)


# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "wb")
f.write(output_graph_def.SerializeToString())

# tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)
