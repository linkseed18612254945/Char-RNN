import numpy as np
import pickle
from collections import Counter


class TextConverter:
    def __init__(self, file_path, max_vocab, byte_file=False):
        if byte_file:
            with open(file_path, 'rb') as f:
                self.vocab_set = pickle.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.text = f.read()
            self.vocab_set = self.__get_vocab(max_vocab)
        self.char_to_int_dict = {c: i for i, c in enumerate(self.vocab_set)}
        self.int_to_char_dict = dict(enumerate(self.vocab_set))
        self.encoded_text = np.array(list(map(lambda x: self.char_to_int_dict[x], self.text)))

    @property
    def vocab_size(self):
        return len(self.vocab_set) + 1

    def __get_vocab(self, max_vocab):
        vocab = set(self.text)
        if len(vocab) > max_vocab > 0:
            vocab = self.__choose_char(self.text, max_vocab)
        return vocab

    def __choose_char(self, text, max_vocab):
        return set(map(lambda x: x[0], Counter(text).most_common(max_vocab)))

    def save_vocab(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.vocab_set, f)

    def batch_generator(self, batch_size, seq_size):
        batch_char_size = batch_size * seq_size
        num_batches = int(len(self.text) / batch_char_size)
        reshape_text = self.encoded_text[:num_batches * batch_char_size].reshape((-1, seq_size))
        np.random.shuffle(reshape_text)
        for i in range(0, reshape_text.shape[0], batch_size):
            input_text = reshape_text[i: i + batch_size, :]
            targets = np.zeros_like(input_text)
            targets[:, :-1], targets[:, -1] = input_text[:, 1:], input_text[:, 0]
            yield input_text, targets


if __name__ == '__main__':
    test_path = './data/poetry.txt'
    tc = TextConverter(test_path, -1)
    batch_data = tc.batch_generator(6, 5)