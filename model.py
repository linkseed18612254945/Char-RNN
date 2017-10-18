import tensorflow as tf
import numpy as np
import os
import time


class CharRNN:
    def __init__(self, batch_size, seq_size, lstm_size, output_size,
                 num_layers=1, train_keep_prob=0.5, learning_rate=0.001, grad_clip=5):
        """
        charRNN模型类
        :param batch_size: 一个batch包含的序列数
        :param seq_size: 每个序列的字符长度
        :param lstm_size: lstm中间层的维度
        :param output_size: 最终输出的结果的维度,通常为训练文本词表的大小
        :param num_layers: lstm节点层数
        :param train_keep_prob: 用于设定节点dropout的比率
        :param learning_rate: 优化器学习速率
        :param grad_clip: 梯度裁剪参数
        """
        self.batch_size = batch_size
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.train_keep_prob = train_keep_prob
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        tf.reset_default_graph()
        self.__build_inputs_ops()
        self.__build_cells()
        self.__build_outputs()
        self.__build_loss()
        self.train_op = self.__build_optimizer()
        self.global_init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.saver = tf.train.Saver()

    def __build_inputs_ops(self):
        """ 创建计算图中的输入节点 """
        with tf.name_scope('input_ops'):
            self.input = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.seq_size), name='input')
            self.one_hot_input = tf.one_hot(self.input, self.output_size, name='one_hot_input')
            self.target = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.seq_size), name='target')
            self.one_hot_target = tf.one_hot(self.target, self.output_size, name='one_hot_target')
            self.reshape_target = tf.reshape(self.one_hot_target, shape=(self.batch_size * self.seq_size, self.output_size))
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    def __build_cells(self):
        """ 构建LSTM层, 包括单个节点和符合节点, 设定cell初始状态 """
        def lstm_cell():
            single_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            single_drop_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(single_lstm_cell, output_keep_prob=self.keep_prob)
            return single_drop_lstm_cell

        with tf.name_scope('lstm_cells'):
            self.cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(self.num_layers)])
            self.initial_state = self.cells.zero_state(self.batch_size, dtype=tf.float32)

    def __build_outputs(self):
        """ 创建计算图中LSTM层的输出状态, 以及变换后的输出结果"""
        with tf.name_scope('output_ops'):
            self.lstm_state_outputs, self.final_state_output = tf.nn.dynamic_rnn(self.cells,
                                                                                 self.one_hot_input,
                                                                                 initial_state=self.initial_state)
            self.reshape_state_outputs = tf.reshape(self.lstm_state_outputs,
                                                    shape=(self.batch_size * self.seq_size, self.lstm_size),
                                                    name='reshape_state_outputs')
            softmax_w = tf.Variable(dtype=tf.float32, initial_value=tf.truncated_normal((self.lstm_size, self.output_size), stddev=0.1))
            softmax_b = tf.Variable(dtype=tf.float32, initial_value=tf.zeros(self.output_size))
            self.logits = tf.matmul(self.reshape_state_outputs, softmax_w) + softmax_b
            self.prob_predict = tf.nn.softmax(self.logits)

    def __build_loss(self):
        """ 创建计算图中损失函数节点, 通常选择softmax后的交叉熵函数 """
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.reshape_target, name='loss')
        self.mean_loss = tf.reduce_mean(self.loss, name='mean_loss')

    def __build_optimizer(self):
        """ 创建计算图中的优化器, 可以选择sgd, adam等优化器并可以进行梯度裁剪, 返回最终的优化训练节点 """
        tvars = tf.trainable_variables()
        gd_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        adam_opt = tf.train.AdamOptimizer(self.learning_rate)
        self.grads = tf.gradients(self.mean_loss, tvars)
        self.clip_grads, _ = tf.clip_by_global_norm(self.grads, self.grad_clip)
        return adam_opt.apply_gradients(zip(self.clip_grads, tvars))

    def train(self, batch_generator, max_steps, model_save_path, save_with_steps, log_with_steps):
        """
        模型训练函数
        :param batch_generator: 训练数据的batch生成器
        :param max_steps:训练的最大步数
        :param model_save_path:模型保存路径
        :param save_with_steps:保存中间模型的间隔步数
        :param log_with_steps:保存日志的间隔步数
        :return:
        """
        total_time_cost = 0
        total_step = 0

        with self.session as sess:
            sess.run(self.global_init)
            for step, batch_data in enumerate(batch_generator):
                total_step = step
                x_batch = batch_data[0]
                y_batch = batch_data[1]
                start = time.time()
                feed = {self.input: x_batch,
                        self.target: y_batch,
                        self.keep_prob: self.train_keep_prob}
                batch_loss, new_state, _ = sess.run([self.mean_loss, self.final_state_output, self.train_op],
                                                    feed_dict=feed)
                end = time.time()
                total_time_cost += end - start
                if step % save_with_steps == 0:
                    self.saver.save(sess, os.path.join(model_save_path, 'model'), global_step=step)
                if step % log_with_steps == 0:
                    training_log_info = 'step: {}/{}...'.format(step, max_steps) + '\n' + \
                                        'loss: {:.4f}'.format(batch_loss) + '\n' + \
                                        'speed: {:.4f} sec/batch'.format(end - start) + '\n' + \
                                        'total time cost: {:.4f}'.format(total_time_cost)
                    print(training_log_info)
                if step >= max_steps > 0:
                    break
            self.saver.save(sess, os.path.join(model_save_path, 'model'))
            complete_log_info = 'Training Complete! {} batches data have been trained. Total time cost: {:.4f}'.format(total_step, total_time_cost)
            print(complete_log_info)

    def sample(self, prime, vocab_size):
        new_state = self.session.run(self.initial_state)
        preds = np.ones(vocab_size)

    def load(self, checkpoin):
        """ 读取保存的模型 """
        self.saver.restore(self.session, checkpoin)
        load_log_info = 'Load model from {}'.format(checkpoin)
        print(load_log_info)

    def test(self, batch_generator, n):
        """ 测试 """
        self.session.run(self.global_init)
        for step, batch_data in enumerate(batch_generator):
            x_batch = batch_data[0]
            y_batch = batch_data[1]
            feed = {self.input: x_batch,
                    self.target: y_batch,
                    self.keep_prob: self.train_keep_prob}
            res = self.session.run(self.prob_predict, feed_dict=feed)
            print(res)
            if step > n:
                break






