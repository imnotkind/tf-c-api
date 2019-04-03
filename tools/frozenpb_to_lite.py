import tensorflow as tf

graph_def_file = "model/mymodel.pb"
input_arrays = ["input/Placeholder", "input/Placeholder_2"]
output_arrays = ["output/ArgMax"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()

open("model/mymodel.tflite", "wb").write(tflite_model)