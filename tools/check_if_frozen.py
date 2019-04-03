import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import sys

f = gfile.FastGFile(sys.argv[1], 'rb')

sess = tf.Session()
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
sess.graph.as_default()
tf.import_graph_def(graph_def)

graph = tf.get_default_graph()

variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

print(variables)

#not working, I don't know why