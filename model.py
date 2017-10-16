import tensorflow as tf
import numpy as np
import os

class CharRNN:
    """

    """
    def __init__(self, num_seqs, num_steps):

        self.num_seqs = num_seqs
        self.num_steps = num_steps

    def build_inputs(self):
        with tf.name_scope('inputs_op'):
            self.inputs = tf.placeholder(tf.int32, shape=(
                self.num_seqs, self.num_steps
            ))