import tensorflow as tf
import util
from model import CharRNN
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('name', 'default_model', 'name of the model')
tf.app.flags.DEFINE_integer('batch_size', 100, 'number of seqs in one batch')
tf.app.flags.DEFINE_integer('seq_size', 100, 'length of one seq')
tf.app.flags.DEFINE_integer('lstm_size', 128, 'size of hidder layer of lstm')
tf.app.flags.DEFINE_integer('num_layers', 2, 'number of layers of lstm')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.app.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout keep rate during training')
tf.app.flags.DEFINE_integer('max_steps', -1, 'the number of batches to use, default is all batches will be used')
tf.app.flags.DEFINE_integer('save_every_n_steps', 1000, 'save the model every n steps')
tf.app.flags.DEFINE_integer('log_every_n_steps', 10, 'print and save the log every n steps')
tf.app.flags.DEFINE_integer('max_vocab', -1, 'max char vocab size, default is all')
tf.app.flags.DEFINE_string('input_file_path', '', 'path of the utf-8 encoded training text')


def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(FLAGS.input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tc = util.TextConverter(text, FLAGS.max_vocab)
    tc.save_vocab(os.path.join('vocab', FLAGS.name))
    output_size = tc.vocab_size
    batch_generator = util.batch_generator(tc.text_to_arr(text), FLAGS.batch_size, FLAGS.seq_size)
    model = CharRNN(output_size=output_size,
                    batch_size=FLAGS.batch_size,
                    seq_size=FLAGS.seq_size,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob)
    model.train(batch_generator,
                max_steps=FLAGS.max_steps,
                model_save_path=model_path,
                save_with_steps=FLAGS.save_every_n_steps,
                log_with_steps=FLAGS.log_every_n_steps)


if __name__ == '__main__':
    tf.app.run()
