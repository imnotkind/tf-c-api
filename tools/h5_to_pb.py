from keras import backend as K
from keras.models import load_model
import tensorflow as tf

model = load_model('model/keras/my_model.h5')

#print(model.summary())  (None,300,300,3) -> (None, 5)
print(model.input)
print(model.output)
print(model.targets)
#print(dir(model))
#print(K.learning_phase())
K.set_learning_phase(0) #0 : test, 1 : train
#print(K.learning_phase())

sess = K.get_session()

saver = tf.train.Saver()
saver.save(sess, 'model/keras/keras.ckpt')

sess.graph.as_default()
graph = sess.graph


saver_def = saver.as_saver_def()
print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

with open('model/keras/keras.pb', 'wb') as f:
    f.write(graph.as_graph_def().SerializeToString())