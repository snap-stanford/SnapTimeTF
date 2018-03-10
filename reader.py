import os
import glob
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange

tf.app.flags.DEFINE_integer("rnn_timesteps", 10, "Number of timesteps used in rnn.")
tf.app.flags.DEFINE_string("data_folder", '/dfs/scratch0/mvc/test/snap_tf/full_split_bool', 'Train tfrecord folder')
tf.app.flags.DEFINE_string("meta_file", 'meta_counts.pkl', "pickle file for metadata on tfrecords")
tf.app.flags.DEFINE_bool("split_bool", True, "If we separate bools from floats in example")
tf.app.flags.DEFINE_bool("shuffle", True, "Whether to shuffle examples")

FLAGS = tf.app.flags.FLAGS

class Reader(object):
    """Takes care of reading from tfrecords using Dataset API"""
    def __init__(self, data_folder=None):
        if data_folder is None:
            data_folder = FLAGS.data_folder
        train_folder, val_folder = [os.path.join(data_folder, s) for s in ('train', 'val')]
        self.train_records, self.val_records =\
            [glob.glob(f + '/*.tfrecord') for f in (train_folder, val_folder)]

        self.shapes = None

    def get_base_feature(self):
        if FLAGS.split_bool:
            return {
                'floats': tf.VarLenFeature(tf.float32),
                'bools': tf.VarLenFeature(tf.float32),
                'num_floats': tf.FixedLenFeature([], tf.int64),
                'num_bools': tf.FixedLenFeature([], tf.int64),
                'num_timesteps': tf.FixedLenFeature([], tf.int64),
            }
        else:
            return {
                'values': tf.VarLenFeature(tf.float32),
                'num_sensors': tf.FixedLenFeature([], tf.int64),
                'num_timesteps': tf.FixedLenFeature([], tf.int64),
            }

    def get_values(self, features):
        if FLAGS.split_bool:
            return [tf.cast(features['bools'], tf.float32), tf.cast(features['floats'], tf.float32)]
        else:
            return [tf.cast(features['values'], tf.float32)]

    def get_sizes(self, features):
        if FLAGS.split_bool:
            return tf.stack([tf.cast(features['num_bools'], tf.int64), tf.cast(features['num_floats'], tf.int64)])
        else:
            return tf.stack([tf.cast(features['num_sensors'], tf.int64)])

    def read_and_decode(self, filename_queue, num_examples=1):
        '''Deprecated, currently only used to get shapes ahead of time'''
        reader = tf.TFRecordReader()
        _, queue_batch = reader.read_up_to(filename_queue, num_examples)
        batch_example = tf.train.shuffle_batch(
            [queue_batch],
            capacity=5,
            num_threads=1,
            batch_size=num_examples,
            min_after_dequeue=2,
            enqueue_many=True,
        )
        
        features = tf.parse_example(
            batch_example,
            features=self.get_base_feature(),
        )

        values = self.get_values(features)
        sizes = self.get_sizes(features)
        num_timesteps = tf.cast(features['num_timesteps'], tf.int64)
        return values, sizes, num_timesteps


    def get_shapes(self, cache=True):
        filename_queue = tf.train.string_input_producer([self.train_records[0]])
        values, sizes, num_timesteps = self.read_and_decode(filename_queue, 1)
        coord = tf.train.Coordinator()
        # hack to get the number of sensors and timestamps before building the rest of the graph
        with tf.Session() as sess:
            print("Analyzing tfrecord data shapes...")
            sess.run(tf.global_variables_initializer())
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            batch_sensor_counts, timestep_count = sess.run([sizes, num_timesteps[0]])
            sensor_counts = np.array(batch_sensor_counts)[:,0]
            print("{0} sensors for {1} timesteps".format(sensor_counts, timestep_count))

        tf.reset_default_graph()
        if cache:
            self.shapes = [(timestep_count, c) for c in sensor_counts]
        return sensor_counts, timestep_count


    def parse_example(self, serialized_example, shapes=None):
        features = tf.parse_single_example(
            serialized_example,
            features=self.get_base_feature()
        )
        values = self.get_values(features)
        sizes = self.get_sizes(features)
        num_timesteps = tf.cast(features['num_timesteps'], tf.int64)
        if shapes is None:
            shapes = self.shapes
        
        dense_values = [tf.reshape(tf.sparse_tensor_to_dense(v, default_value=0), shape) for v, shape in zip(values, shapes)]
        concat_values = tf.concat(dense_values, axis=-1)
        framed_values = tf.contrib.signal.frame(concat_values[:-1], FLAGS.rnn_timesteps, 1, axis=0)
        return concat_values, framed_values, sizes, num_timesteps


    def dataset_batch(self, filenames, batch_size, parallel=32, buffer_size=1000, shape=None):
        if FLAGS.shuffle:
            dataset = tf.data.Dataset.from_tensor_slices(filenames).interleave(tf.data.TFRecordDataset, cycle_length=len(filenames))
        else:
            dataset = tf.data.TFRecordDataset(filenames)
        # dataset = tf.cond(shuffle, lambda: dataset.shuffle(buffer_size=10), lambda: dataset)
        # can't do the above commented line because they come back with different types
        
        parser = lambda ex: self.parse_example(ex, shape)
        dataset = dataset.map(parser, num_parallel_calls=parallel)
        if FLAGS.shuffle:
            # dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size))
        else:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=buffer_size)
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


    def read(self, batch_size, val, shape=None):
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

