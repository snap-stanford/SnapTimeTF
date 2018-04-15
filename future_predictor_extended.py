import tensorflow as tf
import pickle

from model import *

tf.app.flags.DEFINE_string("hidden_layers", '', 'Comma separated hidden layer sizes')
tf.app.flags.DEFINE_integer("rep_index", 0, "Which dense layer to use as the representation. 0 is the output from the rnn")
tf.app.flags.DEFINE_integer("rnn_units", 128, "Number of GRU units to use")
tf.app.flags.DEFINE_bool("bidir_rnn", True, "Whether to use a bidirectional rnn")
tf.app.flags.DEFINE_bool("stack_cells", True, "Whether to use stacked rnn cells")
tf.app.flags.DEFINE_bool("use_cell_state", False, "Whether to use last rnn state for prediction (otherwise, uses output from rnn)")
tf.app.flags.DEFINE_integer("future_timestep", 10, "What number timestep in the future to try to predict")

tf.app.flags.DEFINE_bool("extract", False, "If we just want to extract embeddings")
tf.app.flags.DEFINE_string("embedding_output", 'embed_64.pkl', 'Where to store the embeddings')
tf.app.flags.DEFINE_string("future_loss_keys", '', 'which future losses to include, blank for all')
tf.app.flags.DEFINE_string("opt_on", 'base,future,1', 'which parts to optimize on')


FLAGS = tf.flags.FLAGS


class VWFutureModel(Model):
    """RNN predictions using stacked GRUs"""
    def __init__(self, flags):
        self.rnn_timesteps = flags.rnn_timesteps
        super(VWFutureModel, self).__init__(flags)

    def build_model(self):
        dense_values, frame_values = self.dense_values, self.frame_values

        def cell(num_units):
            if self.flags.stack_cells:
                cell1 = tf.nn.rnn_cell.GRUCell(num_units=num_units)
                cell2 = tf.nn.rnn_cell.GRUCell(num_units=num_units)
                return tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
            else:
                return tf.nn.rnn_cell.GRUCell(num_units=num_units)

        def forward_gru(inputs, num_units, scope=None):
            with tf.variable_scope(scope or "stacked_gru", reuse=tf.AUTO_REUSE):
                outputs, final_state = tf.nn.dynamic_rnn(cell(num_units), inputs=inputs, dtype=tf.float32)
            return outputs, final_state

        def bidir_gru(inputs, num_units, scope=None):
            with tf.variable_scope(scope or "bidir_stacked_gru", reuse=tf.AUTO_REUSE):
                stacked_cell = cell(num_units)
                outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cell, cell_bw=stacked_cell, inputs=inputs, dtype=tf.float32)
            
            final_outputs = outputs[0][:,-1]
            return final_outputs, final_state[0]


        if self.flags.future_timestep > 1:
            frame_values = frame_values[:,:-self.flags.future_timestep+1]


        reshape_frame = lambda x: tf.reshape(x, [-1, frame_values.get_shape()[1], frame_values.get_shape()[-1]])

        num_units = self.flags.rnn_units
        with tf.variable_scope("rnn_with_dense", reuse=tf.AUTO_REUSE):
            def predict(time_values):
                rnn_func = forward_gru
                if self.flags.bidir_rnn:
                    rnn_func = bidir_gru
                outputs, state = rnn_func(time_values, num_units)

                rnn_outputs = state # TODO: state[-1]?
                if not self.flags.use_cell_state:
                    rnn_outputs = outputs

                target = rnn_outputs
                self.rnn_outputs = rnn_outputs
                hidden_sizes = self.flags.hidden_layers.split(',')
                layers = [rnn_outputs]

                if len(hidden_sizes) > 1 or hidden_sizes[0]:
                    hidden_sizes = map(int, hidden_sizes)
                    
                    for i, num_hidden in enumerate(hidden_sizes):
                        with tf.variable_scope("dense{}".format(i)):
                            layers.append(tf.layers.dense(layers[-1], num_hidden, activation=tf.nn.elu, reuse=tf.AUTO_REUSE))

                with tf.variable_scope('predict_1'):
                    predicted = tf.layers.dense(layers[-1], time_values.get_shape().as_list()[-1], reuse=tf.AUTO_REUSE)
                
                target = layers[self.flags.rep_index]
                return predicted, target

            reshape_shape = tf.stack([-1, frame_values.get_shape()[-2], frame_values.get_shape()[-1]])
            reshaped_frames = tf.reshape(frame_values, reshape_shape)
            self.frames = tf.placeholder_with_default(reshaped_frames, shape=(None, self.rnn_timesteps, self.sensor_count))
            predictions, self.compressed = predict(self.frames)
            self.compressed_reshaped = reshape_frame(self.compressed)
            self.compact_predictions = predictions
            self.predicted = reshape_frame(predictions)
            self.expected = dense_values[:,self.rnn_timesteps+self.flags.future_timestep-1:]


        with tf.variable_scope('future'):
            pred_count = self.predicted.get_shape().as_list()[1]
            self.future_predictions = {}
            for k, v in self.future.items():
                with tf.variable_scope(k):
                    dense_prediction = tf.layers.dense(self.compressed, self.frames.get_shape().as_list()[-1])
                    self.future_predictions[k] = {
                        'predicted': reshape_frame(dense_prediction),
                        'expected': v[:,:pred_count]
                    }


        self.future_losses = {k: tf.losses.mean_squared_error(v['expected'], v['predicted']) for k, v in self.future_predictions.items()}
        self.future_loss = sum(self.future_losses.values())
        if self.flags.future_loss_keys:
            loss_keys = self.flags.future_loss_keys.split(',')
            self.future_loss = sum([self.future_losses[k] for k in loss_keys])
        

    def setup_losses(self):
        super(VWFutureModel, self).setup_losses()
        self.orig_loss = self.loss
        opt_params = set(FLAGS.opt_on.split(','))
        if '1' not in opt_params:
            self.loss = 0.0

        if 'future' in opt_params:
            self.loss += self.future_loss

    def setup_opt(self):
        super(VWFutureModel, self).setup_opt()
        future_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'future')
        self.future_opt = tf.train.AdamOptimizer().minimize(self.future_loss, var_list=future_vars)
        t1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'rnn_with_dense/predict_1')
        self.t1_opt = tf.train.AdamOptimizer().minimize(self.orig_loss, var_list=t1_vars)


    def metrics(self):
        d = super(VWFutureModel, self).metrics()
        d.update({
            'orig': self.orig_loss,
            'loss': self.loss,
        })

        d.update(self.future_losses)
        return d

    def metrics_desc(self, args):
        return ', '.join(['{0}: {1:0.3f}'.format(k.split('_')[-1], v) for k, v in args.items()])

    def train_future(self):
        self.setup_session()
        outer = trange(self.flags.epochs)
        steps = self.train_steps

        for i in outer:
            inner = trange(steps)
            for j in inner:
                losses, opt = self.future_losses, self.future_opt
                if '1' in self.flags.opt_on:
                    losses, opt = {'loss': self.loss, 'bool_xent': self.bool_loss, 'float_l2': self.float_loss}, self.t1_opt
                loss_dict, _ = self.sess.run([losses, opt])
                desc = '' + ', '.join(['{0}: {1:0.3f}'.format(k, v) for k, v in loss_dict.items()])
                inner.set_description(desc)

                if j != 0 and j % 100 == 0:
                    self.saver.save(self.sess, self.flags.save_path)

            self.saver.save(self.sess, self.flags.save_path)
            self.writer.flush()


    def predict(self, frames):
        if self.sess is None:
            self.setup_session()

        predict = self.sess.run(self.compact_predictions, feed_dict={self.frames: frames})
        return predict

    def extract_rep(self):
        self.setup_session()
        steps = self.test_steps if self.flags.validate else self.train_steps
        progress = trange(steps)
        fd = {self.val: self.flags.validate}
        timesteps, embeddings = [], []
        for i in progress:
            ts, embed = self.sess.run([self.extra_args['ts'], self.compressed])
            timesteps.extend(ts.tolist())
            embeddings.extend(np.split(embed, self.flags.batch_size))
        return timesteps, embeddings
        

def main(_):
    model = VWFutureModel(FLAGS)
    if FLAGS.extract:
        if FLAGS.driver_file:
            model.extract_rep()
        else:
            reader = model.reader
            d = {}
            
            try:
                for fpath in tqdm(reader.train_records):
                    fname = fpath.split('/')[-1]
                    print(fname)
                    FLAGS.driver_file = fname
                    model = VWFutureModel(FLAGS)
                    ts, embed = model.extract_rep()
                    d[fname] = ts, embed
            except KeyboardInterrupt:
                pass

            with open(FLAGS.embedding_output, 'wb+') as handle:
                pickle.dump(d, handle)

    else:
        opt_params = set(FLAGS.opt_on.split(','))
        if 'base' not in opt_params:
            model.train_future()
        else:
            model.train()

if __name__ == '__main__':
    tf.app.run()
