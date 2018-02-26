import tensorflow as tf

from model import Model

FLAGS = tf.app.flags.FLAGS

class Baseline(Model):
    def __init__(self, flags):
        self.rnn_timesteps = flags.rnn_timesteps
        super(Baseline, self).__init__(flags)

    def build_model(self):
        dense_values = self.dense_values
        self.expected = dense_values[:,self.rnn_timesteps:]
        prev_timestep = dense_values[:,self.rnn_timesteps-1:-1]
        prev_booleans, prev_floats = prev_timestep[:,:,:self.bool_count], prev_timestep[:,:,self.bool_count:]
                
        self.mult = tf.get_variable('mult', shape=(self.bool_count, ), dtype=tf.float32, initializer=tf.ones_initializer())
        self.bias = tf.get_variable('bias', shape=(self.bool_count, ), dtype=tf.float32, initializer=tf.zeros_initializer())
        self.predicted = tf.concat([(prev_booleans+self.bias)*self.mult, prev_floats], axis=-1)


def main(_):
    model = Baseline(FLAGS)
    model.train()
    avg_metrics, predicted, expected = model.validate()
    print avg_metrics
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    tf.app.run()
