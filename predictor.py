import tensorflow as tf

from model import Model

tf.app.flags.DEFINE_string("hidden_layers", '256', 'Comma separated hidden layer sizes')
tf.app.flags.DEFINE_integer("rnn_units", 256, "Number of GRU units to use")
tf.app.flags.DEFINE_bool("bidir_rnn", False, "Whether to use a bidirectional rnn")
tf.app.flags.DEFINE_integer("future_timestep", 10, "What number timestep in the future to try to predict")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "learning rate during training.")

FLAGS = tf.app.flags.FLAGS


class VWModel(Model):
    """RNN predictions using stacked GRUs"""
    def __init__(self, flags):
        self.rnn_timesteps = flags.rnn_timesteps
        super(VWModel, self).__init__(flags)

    def build_model(self):
        dense_values, frame_values = self.dense_values, self.frame_values

        def stacked_lstm(inputs, num_units, scope=None):
            with tf.variable_scope(scope or "stacked_gru", reuse=tf.AUTO_REUSE):
                cell1 = tf.nn.rnn_cell.GRUCell(num_units=num_units)
                cell2 = tf.nn.rnn_cell.GRUCell(num_units=num_units//2)
                stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
                outputs, final_state = tf.nn.dynamic_rnn(stacked_cell, inputs=inputs, dtype=tf.float32)
            return outputs, final_state

        def bidir_stacked_gru(inputs, num_units, scope=None):
            with tf.variable_scope(scope or "bidir_stacked_gru", reuse=tf.AUTO_REUSE):
                cell1 = tf.nn.rnn_cell.GRUCell(num_units=num_units)
                cell2 = tf.nn.rnn_cell.GRUCell(num_units=num_units//2)
                stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
                outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cell, cell_bw=stacked_cell, inputs=inputs, dtype=tf.float32)
            return outputs, final_state[0]


        if self.flags.future_timestep > 1:
            frame_values = frame_values[:,:-self.flags.future_timestep+1]

        num_units = self.flags.rnn_units
        with tf.variable_scope("simple_model", reuse=tf.AUTO_REUSE):
            def predict(time_values):
                rnn_func = stacked_lstm
                if self.flags.bidir_rnn:
                    rnn_func = bidir_stacked_gru
                _, state = rnn_func(time_values, num_units)
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
            self.expected = dense_values[:,self.rnn_timesteps+self.flags.future_timestep-1:]
    

    def predict(self, frames):
        if self.sess is None:
            self.setup_session()

        predict = self.sess.run(self.compact_predictions, feed_dict={self.frames: frames})
        return predict

def main(_):
    model = VWModel(FLAGS)
    model.validate(compute_results=True, save_val=True)

if __name__ == '__main__':
    tf.app.run()
