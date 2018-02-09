import tensorflow as tf
from reader import *
from predictor import *

reader = Reader()
shape = reader.get_shapes()
val = tf.placeholder_with_default(False, [])
dense_values, frame_values, num_sensors, num_timesteps = reader.read(32, shape=shape, val=val)
predictions = tf.layers.dense(dense_values, shape[0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(predictions)