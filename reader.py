import os
import glob
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange

tf.app.flags.DEFINE_integer("rnn_timesteps", 10, "Number of timesteps used in rnn.")
tf.app.flags.DEFINE_string("data_folder", '/dfs/scratch0/mvc/test/snap_tf/test_future_with_ts', 'Train tfrecord folder')
tf.app.flags.DEFINE_string("meta_file", 'meta_counts.pkl', "pickle file for metadata on tfrecords")
tf.app.flags.DEFINE_string("driver_file", '', "specific file for driver, empty to use all")
tf.app.flags.DEFINE_bool("split_bool", True, "If we separate bools from floats in example")
tf.app.flags.DEFINE_bool("shuffle", True, "Whether to shuffle examples")
tf.app.flags.DEFINE_bool("include_ts", True, "Whether the tfrecords contain a timestamp")
tf.app.flags.DEFINE_bool("use_test", False, "Whether to use test (as opposed to validation) records")

FLAGS = tf.app.flags.FLAGS

class Reader(object):
    """Takes care of reading from tfrecords using Dataset API"""
    def __init__(self, data_folder=None):
        if data_folder is None:
            data_folder = FLAGS.data_folder
        self.future_kv = {}
        future_file_path = os.path.join(FLAGS.data_folder, 'future_meta.pkl')
        if os.path.exists(future_file_path):
            with open(future_file_path, 'rb') as f:
                self.future_kv = pickle.load(f)

        folders = [os.path.join(data_folder, s) for s in ('train', 'val', 'test')]
        train_folder, val_folder, test_folder = folders
        glob_str = '/*.tfrecord'
        if FLAGS.driver_file:
            glob_str = '/{}'.format(FLAGS.driver_file)
        self.train_records, self.val_records, self.test_records =\
            [glob.glob(f + glob_str) for f in folders]

        self.all_records = self.train_records + self.val_records + self.test_records

        self.shapes = None

    def get_base_feature(self):
        if FLAGS.split_bool:
            features = {
                'floats': tf.VarLenFeature(tf.float32),
                'bools': tf.VarLenFeature(tf.float32),
                'num_floats': tf.FixedLenFeature([], tf.int64),
                'num_bools': tf.FixedLenFeature([], tf.int64),
                'num_timesteps': tf.FixedLenFeature([], tf.int64),
            }
        else:
            features = {
                'values': tf.VarLenFeature(tf.float32),
                'num_sensors': tf.FixedLenFeature([], tf.int64),
                'num_timesteps': tf.FixedLenFeature([], tf.int64),
            }

        for k in self.future_kv:
            features[k] = tf.VarLenFeature(tf.float32)

        if FLAGS.include_ts:
            features['ts'] = tf.FixedLenFeature([], tf.string)

        return features

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
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
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
        frame = lambda x: tf.contrib.signal.frame(x[:-1], FLAGS.rnn_timesteps, 1, axis=0)
        framed_values = frame(concat_values)
        future, future_shape = {}, (shapes[0][0], sum(map(lambda t: t[1], shapes)))
        for k in self.future_kv:
            dense = tf.reshape(tf.sparse_tensor_to_dense(features[k], default_value=0), future_shape)
            # framed = frame(dense)
            future[k] = dense
        args = {}
        if FLAGS.include_ts:
            args['ts'] = features['ts']
        return concat_values, framed_values, sizes, num_timesteps, future, args


    def dataset_batch(self, filenames, batch_size, parallel=8, buffer_size=1000, shape=None):
        if FLAGS.shuffle:
            dataset = tf.data.Dataset.from_tensor_slices(filenames).interleave(tf.data.TFRecordDataset, cycle_length=len(filenames))
        else:
            dataset = tf.data.TFRecordDataset(filenames)
        # dataset = tf.cond(shuffle, lambda: dataset.shuffle(buffer_size=10), lambda: dataset)
        # can't do the above commented line because they come back with different types
        
        parser = lambda ex: self.parse_example(ex, shape)
        dataset = dataset.map(parser, num_parallel_calls=parallel)
        if FLAGS.shuffle and False:
            # dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size))
        else:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        results = iterator.get_next()
        return results

    def full_meta_counts(self):
        with open(os.path.join(FLAGS.data_folder, 'meta_counts.pkl'), 'rb') as handle:
            d = pickle.load(handle)
        return d
    

    def meta_counts(self):
        d = self.full_meta_counts()
        sub = lambda x: x.replace('/dfs/scratch0/mvc/test/snap_tf/', '')
        return [sum([d[sub(k)] for k in records]) for records in (self.train_records, self.val_records, self.test_records)]
        # return [sum([v for k, v in d.items() if identifier in k]) for identifier in ('train', 'val', 'test')]


    def batch_meta_counts(self, batch_size):
        return [1+x//batch_size for x in self.meta_counts()]


    def read(self, batch_size, val, filenames=None, shape=None):
        train_batch, val_batch, test_batch = [self.dataset_batch(f, batch_size, shape=shape) for f in (self.train_records, self.val_records, self.test_records)]
        # args: dense_values, frame_values, num_sensors, num_timesteps, future
        val_data = test_batch if FLAGS.use_test else val_batch
        args = tf.cond(val, lambda: val_data, lambda: train_batch)
        if filenames is not None:
            args = self.dataset_batch(filenames, batch_size, shape=shape)
        return args


    def stream(self, batch_size, shape=None, val=False, count=None):
        # returns as tuple: concat_values, framed_values, sizes, num_timesteps, future, args
        val_placeholder = tf.placeholder_with_default(val, [], name='validation')
        args = self.read(batch_size, val_placeholder)
        if count is None:
            count = self.meta_counts()[val^1]
        with tf.Session() as sess:
            for i in trange(count):
                elems = sess.run(args)
                yield elems


def main(_):
    reader = Reader()
    reader.get_shapes()
    elems = reader.stream(1).next()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    tf.app.run()
