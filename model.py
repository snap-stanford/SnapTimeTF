import tensorflow as tf
from tqdm import tqdm, trange

import collections
import numpy as np
import os
import pickle

from reader import Reader

tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train for")
tf.app.flags.DEFINE_bool("restore", True, "Whether to restore from save_path")
tf.app.flags.DEFINE_bool("validate", True, "Whether to run validation")

tf.app.flags.DEFINE_integer("cuda_device", 0, "Which gpu to run on")
tf.app.flags.DEFINE_string('graph', './graphs', 'Where to save graph/tensorboard output')
tf.app.flags.DEFINE_string("save_path", 'save_overfit/bool_norm_large.ckpt', "where to save model weights")


FLAGS = tf.app.flags.FLAGS


class Model(object):
    """Generic model that handles reloading, reading, training, metrics, etc."""
    def __init__(self, flags):
        super(Model, self).__init__()
        self.flags = flags
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.flags.cuda_device)
        self.batch_size = flags.batch_size
        self.reader = Reader()        

        self.sensor_counts, self.timestep_count = self.reader.get_shapes()
        self.sensor_count = sum(self.sensor_counts)
        self.bool_count = self.sensor_counts[0]

        self.train_steps, self.val_steps = self.reader.batch_meta_counts(self.batch_size)
        self.sess = None
        self._setup_model()
        self.build_model()
        self.setup_losses()
        self.setup_tensorboard()


    def build_model(self):
        raise NotImplementedError


    def _setup_model(self):
        self.val = tf.placeholder_with_default(False, [], name='validation')
        self.dense_values, self.frame_values, sensor_counts, num_timesteps = self.reader.read(self.batch_size, val=self.val)
        self.predicted, self.expected = None, None
        # Note: override self.predicted and self.expected in build_model


    def setup_losses(self):
        self.bool_loss = tf.constant(0)
        self.loss = 0.0
        square_loss = tf.squared_difference(self.predicted, self.expected)
        if self.flags.split_bool:
            pred, expect = [(x[:,:,:self.bool_count], x[:,:,self.bool_count:]) for x in (self.predicted, self.expected)]
            square_loss = tf.squared_difference(pred[1], expect[1])
            self.bool_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=expect[0], logits=pred[0]))
            self.predicted = tf.concat([tf.nn.sigmoid(pred[0]), pred[1]], axis=-1)
            self.loss += self.bool_loss

        self.float_loss = tf.reduce_mean(square_loss)
        self.loss += self.float_loss
        self.mse = tf.losses.mean_squared_error(self.expected, self.predicted)
        self.abs_diff = tf.losses.absolute_difference(self.expected, self.predicted)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
        self.signal_error = tf.reduce_mean(tf.reduce_mean(tf.square(self.predicted - self.expected), axis=0), axis=0)


    def setup_tensorboard(self):
        self.summary = self.summary_op()
        self.val_summary = self.val_summary_op()

    def metrics(self):
        return {
            'loss': self.loss,
            'mse': self.mse,
            'l1': self.abs_diff,
            'bool_xent': self.bool_loss,
            'float_l2': self.float_loss,
            'signal_error': self.signal_error
        }

    def summary_op(self):
        with tf.name_scope("train_summary"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("bool_loss", self.bool_loss)
            tf.summary.scalar("mse", self.mse)
            tf.summary.scalar("abs_diff", self.abs_diff)
            return tf.summary.merge_all()

    def val_summary_op(self):
        with tf.name_scope("val_summary"):
            tf.summary.scalar("val_loss", self.loss)
            tf.summary.scalar("val_mse", self.mse)
            tf.summary.scalar("val_abs_diff", self.abs_diff)
            return tf.summary.merge_all()

    def setup_session(self):
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver()

        self.sess = tf.Session(config=config)
        self.writer = tf.summary.FileWriter(self.flags.graph, self.sess.graph)

        if self.flags.restore:
            print("Restoring weights...")
            self.saver.restore(self.sess, self.flags.save_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        self.batch_window_count = self.flags.batch_size*(self.timestep_count-self.flags.rnn_timesteps)

    def discard_session(self):
        self.sess.close()
        tf.reset_default_graph()

    def train(self, steps=None):
        self.setup_session()
        outer = trange(self.flags.epochs)
        
        if steps is None:
            steps = self.train_steps

        metrics = self.metrics()

        for i in outer:
            inner = trange(steps)
            for j in inner:
                measured, summary, _ = self.sess.run([metrics, self.summary, self.opt])
                inner.set_description("Loss: {0:0.5f}, float l2: {1:0.5f}, bool: {2:0.2f}".format(*[measured[k] for k in ['loss', 'float_l2', 'bool_xent']])) 
                self.writer.add_summary(summary, global_step=(i*steps+j)*self.batch_window_count)

                if j != 0 and j % 100 == 0:
                    self.saver.save(self.sess, self.flags.save_path)
                    self.writer.flush()

            self.saver.save(self.sess, self.flags.save_path)
            self.writer.flush()

            if self.flags.validate:
                measured = self.validate(step_offset=i, write_tensorboard=True, compute_results=False)[0]
                self.writer.flush()
                outer.set_description("Val avg loss: {}".format(measured['loss']))

    def validate(self, steps=None, step_offset=0, write_tensorboard=False, compute_results=False, use_validation_set=True, save_val=False):
        if self.sess is None:
            self.setup_session()
        if steps is None:
            steps = self.val_steps if use_validation_set else self.train_steps
        val_tqdm = trange(steps)
        predicted, expected = [], []

        metrics = self.metrics()
        agg_metrics = collections.defaultdict(float)
        fd = {self.val: use_validation_set}

        for j in val_tqdm:
            if compute_results:
                measured, predict_batch, expect_batch = self.sess.run([metrics, self.predicted, self.expected], feed_dict=fd)
                predicted.append(predict_batch)
                expected.append(expect_batch)
            else:
                measured, summary = self.sess.run([metrics, self.val_summary], feed_dict=fd)
            
            for name, value in measured.iteritems():
                agg_metrics[name] += value

            val_tqdm.set_description("Val Loss: {}".format(agg_metrics['loss']/(j+1)))
            if write_tensorboard:
                self.writer.add_summary(summary, global_step=(steps*step_offset + j)*self.batch_window_count)
        
        avg_metrics = {k: v/steps for k, v in agg_metrics.iteritems()}
        
        if save_val:
            pickle.dump(predicted, open('predicted_validate.pkl', 'wb'))
            pickle.dump(expected, open('expected_validate.pkl', 'wb'))
            pickle.dump(metrics, open('avg_metrics_validate.pkl', 'wb'))
        return avg_metrics, predicted, expected
        
    