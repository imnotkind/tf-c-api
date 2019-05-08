import tensorflow as tf


def frozen_fcn():
  GRAPH_PREFIX = "model/frozen_fcn"

  graph_def_file = GRAPH_PREFIX+".pb"
  input_arrays = ["input_image", "keep_probability"]
  input_shapes = {
    "input_image" : [1, 256, 256, 3],
    "keep_probability" : [1], #no scalar value : https://github.com/tensorflow/tensorflow/issues/23932
  }
  output_arrays = ["Pred"]
  

  converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays, input_shapes)
  tflite_model = converter.convert()

  open(GRAPH_PREFIX+".tflite", "wb").write(tflite_model)

if __name__=="__main__":
  frozen_fcn()