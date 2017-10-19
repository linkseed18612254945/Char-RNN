import numpy as np
import pickle
from collections import Counter


class TextConverter:
    def __init__(self, text, max_vocab, byte_file=None):
        """
        文件读取类,可以直接读取文本文件也可以读取byte pickle文件.负责确定文件的字符词表
        完成文件字符和编码数字的映射.
        :param file_path: 文件路径,可以是文本文件也可以是二进制文件
        :param max_vocab: 允许保留的最大词典容量,默认为负值时将全部保留
        :param byte_file: 输入文件是否为二进制文件
        """
        if byte_file is not None:
            with open(byte_file, 'rb') as f:
                self.vocab_set = pickle.load(f)
        else:
            self.vocab_set = self.__get_vocab(text, max_vocab)
        self.char_to_int_dict = {c: i for i, c in enumerate(self.vocab_set)}
        self.int_to_char_dict = dict(enumerate(self.vocab_set))

    @property
    def vocab_size(self):
        return len(self.vocab_set)

    def __get_vocab(self, text, max_vocab):
        vocab = set(text)
        if len(vocab) > max_vocab > 0:
            vocab = self.__choose_char(text, max_vocab)
        return vocab

    def __choose_char(self, text, max_vocab):
        return set(map(lambda x: x[0], Counter(text).most_common(max_vocab)))

    def save_vocab(self, save_path):
        """ 将词表保存为二进制文件 """
        with open(save_path, 'wb') as f:
            pickle.dump(self.vocab_set, f)

    def text_to_arr(self, text):
        arr = []
        for char in text:
            if char in self.char_to_int_dict:
                arr.append(self.char_to_int_dict[char])
            else:
                arr.append(self.char_to_int_dict[' '])
        return np.array(arr)

    def arr_to_text(self, arr):
        return "".join(list(map(lambda x: self.int_to_char_dict[x], arr)))

    def my_batch_generator(self, batch_size, seq_size):
        """ 生成指定batch大小和序列长度的输入数据生成器 """
        encoded_text = np.array(list(map(lambda x: self.char_to_int_dict[x], self.text)))
        batch_char_size = batch_size * seq_size
        num_batches = int(len(self.text) / batch_char_size)
        reshape_text = encoded_text[:num_batches * batch_char_size].reshape((-1, seq_size))
        while True:
            np.random.shuffle(reshape_text)
            for i in range(0, reshape_text.shape[0], batch_size):
                input_text = reshape_text[i: i + batch_size, :]
                targets = np.zeros_like(input_text)
                targets[:, :-1], targets[:, -1] = input_text[:, 1:], input_text[:, 0]
                yield input_text, targets


def batch_generator(arr, batch_size, seq_size):
    batch_chars = batch_size * seq_size
    n_batches = int(len(arr) / batch_chars)
    arr = arr[:batch_chars * n_batches]
    arr = arr.reshape((batch_size, -1))
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], seq_size):
            x = arr[:, n:n + seq_size]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


def choose_class_by_prediction(pred, top_n=5):
    """ 根据概率分布结果, 在概率前n的类别中随机选择一个类别作为最终的输出类别 """
    p = np.squeeze(pred)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    return np.random.choice(p.size, 1, p=p)[0]

