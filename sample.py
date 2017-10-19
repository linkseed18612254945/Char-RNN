import tensorflow as tf
from util import TextConverter
from model import CharRNN
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.app.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.app.flags.DEFINE_string('vocab_path', '', 'the path of vocab')
tf.app.flags.DEFINE_string('start_string', '', 'use the string to start generating')
tf.app.flags.DEFINE_integer('length', 500, 'max length to generate')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'the path of model checkpoint')
tf.app.flags.DEFINE_string('save_path', './output/generate_text.txt', 'the path of generate text')


def main(_):
    tc = TextConverter("", -1, byte_file=FLAGS.vocab_path)
    output_size = tc.vocab_size
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    model = CharRNN(output_size=output_size,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    sampling=True)
    model.load(FLAGS.checkpoint_path)
    start = tc.text_to_arr(FLAGS.start_string)
    generate_arr = model.sample(FLAGS.length, start, output_size)
    generate_text = tc.arr_to_text(generate_arr)
    with open(FLAGS.save_path, 'w', encoding='utf-8') as f:
        f.write(generate_text)
    print(generate_text)


# python3 sample.py --vocab_path ./vocab/shakespear --length 1000 --checkpoint_path ./model/shakespear
if __name__ == '__main__':
    tf.app.run()

