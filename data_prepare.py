import time
import numpy as np
import tensorflow as tf

class TextConverter:
    def __init__(self, text, max_vocab=100, filename=None):
        self.vocab = set(text)
        self.vocab_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_vocab = dict(enumerate(self.vocab))
        self.decode_text = np.array([self.vocab_to_int[c] for c in text], dtype=np.int32)

    def get_decode_text(self):
        return self.decode_text

    def get_vocab_num(self):
        return len(self.vocab)

def batch_generator(arr, num_seqs, num_steps):
    """
    mini-batch
    :param arr:
    :param num_seqs: the number of sequence in a batch
    :param num_steps: the length of a sequence
    :return:
    """

    batch_size = num_seqs * num_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((-1, num_steps))
    #np.random.shuffle(arr)
    for n in range(0, arr.shape[0], num_seqs):
        x = arr[n:n + num_seqs, :]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

def batch_generator_other(arr, n_seqs, n_steps):
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
        #np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

def build_inputs(num_seqs, num_steps):
    inputs = tf.placeholder(dtype=tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(dtype=tf.int32, shape=(num_seqs, num_steps), name='targets')

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    """

    :param lstm_size: hider layer size(state size)
    :param num_layers:
    :param batch_size: num_seqs * num_steps
    :param keep_prob:
    :return:
    """
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    return lstm, cell, initial_state


def build_output(lstm_output, in_size, out_size):
    """
    :param lstm_output:
    :param in_size:
    :param out_size:
    :return:
    """
    seq_output = tf.concat(lstm_output, 1)
    x = tf.reshape(seq_output, [-1, in_size])



if __name__ == '__main__':
    num_seqs = 5
    num_steps = 6
    lstm_size = 64
    num_layers = 1


    with open('./test_data') as f:
        text = f.read()
    tc = TextConverter(text)
    out_size = tc.get_vocab_num()

    text_arr = tc.get_decode_text()
    batch_data = batch_generator_other(text_arr, num_seqs, num_steps)

    inputs, targets, keep_prob = build_inputs(num_seqs, num_steps)
    lstm, cell, initial_state = build_lstm(lstm_size, num_layers, num_seqs, keep_prob)
    x_one_hot = tf.one_hot(inputs, out_size)
    outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)

    # output
    seq_output = tf.concat(outputs, 1)
    x = tf.reshape(seq_output, [-1, lstm_size])
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([lstm_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits)

    # loss
    y_one_hot = tf.one_hot(targets, out_size)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss_mean = tf.reduce_mean(loss)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mean)

    init = tf.global_variables_initializer()



    sess = tf.Session()
    batch = batch_data.__next__()
    x_batch, y_batch = batch[0], batch[1]
    feed = {inputs: x_batch, targets: y_batch}
    sess.run(init)
    sess.run(outputs, feed_dict=feed)

