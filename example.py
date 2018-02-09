import tensorflow as tf
from reader import *
from predictor import *

reader = Reader()
shape = reader.get_shapes()
dense_values, frame_values, num_sensors, num_timesteps = reader.read(32, shape=shape)
predictions = tf.layers.dense(dense_values)

with tf.Session() as sess:
    sess.run(predictions)


