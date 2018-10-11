import tensorflow as tf
import numpy as np
import time
import os


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

class CharRNN(object):
    def __init__(self, num_words, batch_size=64, num_steps=50, n_neurons=128, n_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, embedding=True, embedding_size=128):
        if sampling is True:
            batch_size, num_steps = 1, 1

        self.num_words = num_words
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.sampling = sampling
        self.train_keep_prob = train_keep_prob
        self.embedding = embedding
        self.embedding_size = embedding_size

        # remember to reset the default graph
        tf.reset_default_graph()
        #################################################
        # Model
        # build the input placeholder
        with tf.name_scope("input"):
            self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.num_steps], name='input_sequences')
            self.input_y = tf.placeholder(tf.int32, [self.batch_size, self.num_steps], name='target_sequences')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # if the input is chinese, build an embedding layer
        if self.embedding is False:
            self.lstm_inputs = tf.one_hot(self.input_x, self.num_words)
        else:
            with tf.device("/cpu:0"), tf.name_scope('embedding'):
                W = tf.get_variable("W", [self.num_words, self.embedding_size])
                self.lstm_inputs = tf.nn.embedding_lookup(W, self.input_x)

        with tf.name_scope("lstm"):
            lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_neurons) for _ in range(self.n_layers)]
            cells_drop = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob) for cell in
                          lstm_cells]
            # input_keep_prob
            self.multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells_drop)
            self.initial_state = self.multi_cell.zero_state(self.batch_size, tf.float32)

            # get lstm output and the final state of the model
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(self.multi_cell, self.lstm_inputs,
                                                                    initial_state=self.initial_state)

            # reshape the output / row_len = batch_size * num_steps, column_len = n_neurons
            seq_output = tf.concat(self.lstm_outputs, 1)
            h_seq_output = tf.reshape(seq_output, [-1, self.n_neurons])  # -1 means length unknown

        with tf.name_scope("output"):
            # self.output = tf.layers.dense(h_seq_output, self.num_words, name='output')
            # self.predictions_proba = tf.nn.softmax(self.output, name='predictions')

            w = tf.Variable(tf.truncated_normal([self.n_neurons, self.num_words], stddev=0.1))
            b = tf.Variable(tf.zeros(self.num_words))
            self.output = tf.matmul(h_seq_output, w) + b
            self.predictions_proba = tf.nn.softmax(self.output, name='predictions')

        with tf.name_scope("loss"):
            y_one_hot = tf.one_hot(self.input_y, self.num_words)  # 把y(target sequence)变为one hot的形式
            y_reshaped = tf.reshape(y_one_hot, self.output.get_shape())  # 把y_one_hot reshape为跟output一样的维度
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=y_reshaped)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('train'):
            # 使用clipping gradients
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
            train_op = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = train_op.apply_gradients(zip(grads, tvars))

        self.saver = tf.train.Saver()

    def fit(self, batch_generator, n_iterations, save_path, save_every_n, log_every_n):
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        with self.session as sess:
            sess.run(init)
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.input_x: x,
                        self.input_y: y,
                        self.dropout_keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.optimizer],
                                                    feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, n_iterations),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= n_iterations:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def predict(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size,))  # for prime=[]
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.input_x: x,
                    self.dropout_keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.predictions_proba, self.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(c)

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.input_x: x,
                    self.dropout_keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.predictions_proba, self.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)


    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))