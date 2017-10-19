import util
import numpy as np
import model
import codecs


TEST_DATA_PATH = './data/shakespeare.txt'
TEST_BATCH_SIZE = 32
TEST_SEQ_SIZE = 50
TEST_LSTM_SIZE = 128
TEST_NUM_LAYERS = 2
TEST_RATE = 0.01
TEST_KEEP_PROB = 0.5

def textconverter_test():
    with open(TEST_DATA_PATH, encoding='utf-8') as f:
        text = f.read()
    tc = util.TextConverter(text, -1)
    # 词表转化测试
    print(tc.vocab_size, tc.vocab_set)
    # 编码字典测试
    print(tc.int_to_char_dict)
    print(tc.char_to_int_dict)
    # 字符串与编码向量转换测试
    arr = tc.text_to_arr(text[:500])
    print(arr, arr.shape)
    t = tc.arr_to_text(arr)
    print(t)


def batch_test():
    with open(TEST_DATA_PATH, encoding='utf-8') as f:
        text = f.read()
    tc = util.TextConverter(text, -1)
    g = util.batch_generator(tc.text_to_arr(text), TEST_BATCH_SIZE, TEST_SEQ_SIZE)

    x_batch, y_batch = g.__next__()
    print(x_batch.shape, x_batch)
    for arr in x_batch:
        print(tc.arr_to_text(arr))
    print(y_batch.shape, y_batch)
    for arr in y_batch:
        print(tc.arr_to_text(arr))

def model_test():
    with open(TEST_DATA_PATH, encoding='utf-8') as f:
        text = f.read()
    tc = util.TextConverter(text, -1)
    g = util.batch_generator(tc.text_to_arr(text), TEST_BATCH_SIZE, TEST_SEQ_SIZE)

    # 模型加载测试
    rnn_model = model.CharRNN(output_size=tc.vocab_size,
                    batch_size=TEST_BATCH_SIZE,
                    seq_size=TEST_SEQ_SIZE,
                    lstm_size=TEST_LSTM_SIZE,
                    num_layers=TEST_NUM_LAYERS,
                    learning_rate=TEST_RATE,
                    train_keep_prob=TEST_KEEP_PROB)
    x_batch, y_batch = g.__next__()
    sess = rnn_model.session
    state = sess.run(rnn_model.initial_state)
    feed = {rnn_model.input: x_batch, rnn_model.target: y_batch,
            rnn_model.initial_state: state, rnn_model.keep_prob: TEST_KEEP_PROB}
    # 模型输入流测试
    one_hot_input = sess.run(rnn_model.one_hot_input, feed_dict=feed)
    print(one_hot_input.shape, one_hot_input)



if __name__ == '__main__':
    textconverter_test()