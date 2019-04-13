import tensorflow as tf


def frozen_fcn():
  GRAPH_PREFIX = "model/frozen_fcn"

  graph_def_file = GRAPH_PREFIX+".pb"
  input_arrays = ["input_image", "keep_probabilty"]
  input_shapes = {
    "input_image" : [1, 900, 600, 3],
    "keep_probabilty" : 0
  }
  output_arrays = ["Pred"]
  

  converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays, input_shapes)
  tflite_model = converter.convert()

  open(GRAPH_PREFIX+".tflite", "wb").write(tflite_model)

def mymodel():
  GRAPH_PREFIX = "model/mymodel"

  graph_def_file = GRAPH_PREFIX+".pb"
  input_arrays = ["input/Placeholder", "input/Placeholder_2"]
  input_shapes = {
    "input/Placeholder" : [1, 500, 1024, 1],
    "input/Placeholder_2" : 0
  }
  output_arrays = ["output/ArgMax"]
  

  converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file, input_arrays, output_arrays, input_shapes)
  tflite_model = converter.convert()

  open(GRAPH_PREFIX+".tflite", "wb").write(tflite_model)

if __name__=="__main__":
  mymodel()
  #frozen_fcn()