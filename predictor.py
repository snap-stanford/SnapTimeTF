import tensorflow as tf
from tqdm import tqdm, trange

import numpy as np
import os

from reader import Reader

tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_string("hidden_layers", '256', 'Comma separated hidden layer sizes')
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "learning rate during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train for")
tf.app.flags.DEFINE_string("save_path", 'save/bool_norm_double_dense_model.ckpt', "where to save model weights")
tf.app.flags.DEFINE_bool("restore", True, "Whether to restore from save_path")
tf.app.flags.DEFINE_bool("validate", True, "Whether to run validation")


FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class VWModel(object):
    """docstring for VWModel"""
    def __init__(self, flags):
        self.flags = flags
        self.rnn_timesteps = flags.rnn_timesteps
        self.batch_size = flags.batch_size
        self.reader = Reader()        

        self.sensor_count, self.timestep_count = self.reader.get_shapes()
        self.build_model()
        self.train_steps, self.val_steps = self.reader.batch_meta_counts(self.batch_size)
        self.sess = None

    def build_model(self):
        self.val = tf.placeholder_with_default(False, [], name='validation')
        dense_values, frame_values, num_sensors, num_timesteps = self.reader.read(self.batch_size, shape=(self.timestep_count, self.sensor_count), val=self.val)

        def simple_rnn(inputs, num_units, scope=None):
            with tf.variable_scope(scope or "simple_rnn", reuse=tf.AUTO_REUSE):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
                outputs, _ = tf.nn.dynamic_rnn(cell, inputs=inputs, dtype=tf.float32)
            return outputs, _

        def stacked_lstm(inputs, num_units, scope=None):
            with tf.variable_scope(scope or "stacked_gru", reuse=tf.AUTO_REUSE):
                cell1 = tf.nn.rnn_cell.GRUCell(num_units=num_units)
                cell2 = tf.nn.rnn_cell.GRUCell(num_units=num_units//2)
                stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
                outputs, final_state = tf.nn.dynamic_rnn(stacked_cell, inputs=inputs, dtype=tf.float32)
            return outputs, final_state


        num_units = 128
        use_chunking = False
        use_frames = True
        with tf.variable_scope("simple_model", reuse=tf.AUTO_REUSE):
            def predict(time_values):
                # time_values, expected = dense_values[:, i:i+self.rnn_timesteps], dense_values[:, i+self.rnn_timesteps+1]
                _, state = stacked_lstm(time_values, num_units)
                hidden_sizes = self.flags.hidden_layers.split(',')
                if len(hidden_sizes) == 1 and not hidden_sizes[0]:
                    predicted =  tf.layers.dense(state[-1], time_values.get_shape().as_list()[-1], reuse=tf.AUTO_REUSE)

                else:
                    hidden_sizes = map(int, hidden_sizes)
                    layers = [state[-1]]
                    for i, num_hidden in enumerate(hidden_sizes):
                        with tf.variable_scope("dense{}".format(i)):
                            layers.append(tf.layers.dense(layers[-1], num_hidden, activation=tf.nn.relu, reuse=tf.AUTO_REUSE))

                    predicted = tf.layers.dense(layers[-1], time_values.get_shape().as_list()[-1], reuse=tf.AUTO_REUSE)

                return predicted

            
            reshape_shape = tf.stack([-1, frame_values.get_shape()[-2], frame_values.get_shape()[-1]])
            reshaped_frames = tf.reshape(frame_values, reshape_shape)
            self.frames = tf.placeholder_with_default(reshaped_frames, shape=(None, self.rnn_timesteps, self.sensor_count))
            predictions = predict(self.frames)
            self.compact_predictions = predictions
            self.predicted = tf.reshape(predictions, [-1, frame_values.get_shape()[1], frame_values.get_shape()[-1]])
            self.expected = dense_values[:,self.rnn_timesteps:]
                
        self.loss_sq = tf.reduce_mean(tf.square(self.predicted - self.expected))
        self.loss = tf.norm(self.predicted - self.expected)
        self.mse = tf.losses.mean_squared_error(self.expected, self.predicted)
        self.abs_diff = tf.losses.absolute_difference(self.expected, self.predicted)
        # self.rmse = tf.metrics.root_mean_squared_error(self.expected, self.predicted)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
        self.summary = self.summary_op()
        self.val_summary = self.val_summary_op()

    def summary_op(self):
        with tf.name_scope("train_summary"):
            tf.summary.scalar("norm_loss", self.loss)
            tf.summary.scalar("loss_sq", self.loss_sq)
            tf.summary.scalar("mse", self.mse)
            # tf.summary.scalar("rmse", self.rmse)
            tf.summary.scalar("abs_diff", self.abs_diff)
            return tf.summary.merge_all()

    def val_summary_op(self):
        with tf.name_scope("val_summary"):
            tf.summary.scalar("val_norm_loss", self.loss)
            tf.summary.scalar("val_loss_sq", self.loss_sq)
            tf.summary.scalar("val_mse", self.mse)
            # tf.summary.scalar("val_rmse", self.rmse)
            tf.summary.scalar("val_abs_diff", self.abs_diff)
            return tf.summary.merge_all()

    def setup_session(self):
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # config.gpu_options.visible_device_list="1"
        self.saver = tf.train.Saver()

        self.sess = tf.Session(config=config)
        self.writer = tf.summary.FileWriter('./pca_graphs', self.sess.graph)

        if self.flags.restore:
            print("Restoring weights...")
            self.saver.restore(self.sess, self.flags.save_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        self.batch_window_count = self.flags.batch_size*(self.timestep_count-self.flags.rnn_timesteps)

    def train(self, steps=None):
        self.setup_session()
        outer = trange(self.flags.epochs)
        if steps is None:
            steps = self.train_steps
        for i in outer:
            inner = trange(steps)
            for j in inner:
                loss_value, loss_sq, summary, _ = self.sess.run([self.loss, self.loss_sq, self.summary, self.opt])
                inner.set_description("Loss: {0}, loss sq: {1}".format(loss_value, loss_sq))
                self.writer.add_summary(summary, global_step=(i*steps+j)*self.batch_window_count)
                if j != 0 and j % 100 == 0:
                    self.saver.save(self.sess, self.flags.save_path)
                    self.writer.flush()

            self.saver.save(self.sess, self.flags.save_path)
            self.writer.flush()

            if self.flags.validate:
                val_mse = self.validate(step_offset=i, compute_results=False)[0]
                self.writer.flush()
                outer.set_description("Val avg MSE: {}".format(val_mse))
            

    def validate(self, steps=None, step_offset=0, write_tensorboard=False, compute_results=True):
        if self.sess is None:
            self.setup_session()
        val_mse_total = 0.0
        if steps is None:
            steps = self.val_steps
        val_tqdm = trange(steps)
        predicted, expected = [], []
        for j in val_tqdm:
            fd = {self.val: True}
            if compute_results:
                mse_value, predict_batch, expect_batch = self.sess.run([self.loss_sq, self.predicted, self.expected], feed_dict=fd)
                predicted.append(predict_batch)
                expected.append(expect_batch)
            else:
                mse_value, summary = self.sess.run([self.loss_sq, self.val_summary], feed_dict=fd)
            
            val_mse_total += mse_value
            val_tqdm.set_description("Val Loss: {}".format(val_mse_total/(j+1)))
            if write_tensorboard:
                self.writer.add_summary(summary, global_step=(steps*step_offset + j)*self.batch_window_count)
        
        return val_mse_total/steps, predicted, expected

    def predict(self, frames):
        if self.sess is None:
            self.setup_session()

        # for batch in tqdm(np.array_split(frames, len(frames)//self.flags.batch_size)):
        predict = self.sess.run(self.compact_predictions, feed_dict={self.frames: frames})
        return predict

def main(_):
    model = VWModel(FLAGS)
    model.train()

if __name__ == '__main__':
    tf.app.run()
