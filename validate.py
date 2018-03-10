import tensorflow as tf

from predictor import VWModel

FLAGS = tf.flags.FLAGS

def main(_):
    model = VWModel(FLAGS)
    vw_train = model.validate(use_validation_set=False)
    vw_val = model.validate(use_validation_set=True)
    print('train', vw_train)
    print('val', vw_val)

if __name__ == '__main__':
    tf.app.run()