import os
import glob
import pickle

import tensorflow as tf
from tqdm import tqdm, trange

tf.app.flags.DEFINE_integer("rnn_timesteps", 10, "Number of timesteps used in rnn.")
tf.app.flags.DEFINE_string("data_folder", 'bool_norm', 'Train tfrecord folder')
tf.app.flags.DEFINE_string("meta_file", 'meta_counts.pkl', "pickle file for metadata on tfrecords")

FLAGS = tf.app.flags.FLAGS

class Reader(object):
    """Takes care of reading from tfrecords using Dataset API"""
    def __init__(self, data_folder=None):
        if data_folder is None:
            data_folder = FLAGS.data_folder
        train_folder, val_folder = [os.path.join(data_folder, s) for s in ('train', 'val')]
        self.train_records, self.val_records =\
            [glob.glob(f + '/*.tfrecord') for f in (train_folder, val_folder)]


    def read_and_decode(self, filename_queue, num_examples):
        '''Deprecated, currently only used to get shapes ahead of time'''
        reader = tf.TFRecordReader()
        _, queue_batch = reader.read_up_to(filename_queue, num_examples)
        batch_example = tf.train.shuffle_batch(
            [queue_batch],
            capacity=5000,
            num_threads=8,
            batch_size=num_examples,
            min_after_dequeue=500,
            enqueue_many=True,
        )
        features = tf.parse_example(
            batch_example,
            features={
                'values': tf.VarLenFeature(tf.float32),
                'num_sensors': tf.FixedLenFeature([], tf.int64),
                'num_timesteps': tf.FixedLenFeature([], tf.int64),
            }
        )

        values = tf.cast(features['values'], tf.float32)
        num_sensors = tf.cast(features['num_sensors'], tf.int64)
        num_timesteps = tf.cast(features['num_timesteps'], tf.int64)
        return values, num_sensors, num_timesteps


    def get_shapes(self):
        filename_queue = tf.train.string_input_producer([self.train_records[0]])
        values, num_sensors, num_timesteps = self.read_and_decode(filename_queue, 1)
        coord = tf.train.Coordinator()
        # hack to get the number of sensors and timestamps before building the rest of the graph
        with tf.Session() as sess:
            print("Analyzing tfrecord data shapes...")
            sess.run(tf.global_variables_initializer())
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sensor_count, timestep_count = sess.run([num_sensors[0], num_timesteps[0]])
            print("{0} sensors for {1} timesteps".format(sensor_count, timestep_count))

        tf.reset_default_graph()
        return sensor_count, timestep_count


    def parse_example(self, serialized_example, shape=None):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'values': tf.VarLenFeature(tf.float32),
                'num_sensors': tf.FixedLenFeature([], tf.int64),
                'num_timesteps': tf.FixedLenFeature([], tf.int64),
            }
        )
        values = tf.cast(features['values'], tf.float32)
        num_sensors = tf.cast(features['num_sensors'], tf.int64)
        num_timesteps = tf.cast(features['num_timesteps'], tf.int64)
        if shape is None:
            shape = tf.stack([num_timesteps, num_sensors])
        dense_values = tf.sparse_tensor_to_dense(values, default_value=0)
        dense_values = tf.reshape(dense_values, shape)
        framed_values = tf.contrib.signal.frame(dense_values[:-1], FLAGS.rnn_timesteps, 1, axis=0)
        return dense_values, framed_values, num_sensors, num_timesteps


    def dataset_batch(self, filenames, batch_size, parallel=16, shape=None):
        dataset = tf.data.TFRecordDataset(filenames).repeat()
        parser = lambda ex: self.parse_example(ex, shape)
        dataset = dataset.map(parser, num_parallel_calls=parallel)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        results = iterator.get_next()
        return results
    

    def meta_counts(self):
        with open(os.path.join(FLAGS.data_folder, 'meta_counts.pkl'), 'rb') as handle:
            d = pickle.load(handle)
        train_count, val_count = [sum([v for k, v in d.items() if identifier in k]) for identifier in ('train', 'val')]
        return train_count, val_count


    def batch_meta_counts(self, batch_size):
        return [1+x//batch_size for x in self.meta_counts()]


    def read(self, batch_size, shape=None, val=tf.placeholder_with_default(False, [])):
        train_batch, val_batch = [self.dataset_batch(f, batch_size, shape=shape) for f in (self.train_records, self.val_records)]
        dense_values, frame_values, num_sensors, num_timesteps = tf.cond(val, lambda: val_batch, lambda: train_batch)
        return dense_values, frame_values, num_sensors, num_timesteps


    def stream(self, batch_size, shape=None, val=False, count=None):
        val_placeholder = tf.placeholder_with_default(val, [], name='validation')
        args = self.read(batch_size, shape, val_placeholder)
        if count is None:
            count = self.meta_counts()[val^1]
        with tf.Session() as sess:
            for i in trange(count):
                elems = sess.run(args)
                yield elems

