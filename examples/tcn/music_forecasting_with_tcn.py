"""
This script replicates some of the experiments run in the paper:
Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv preprint arXiv:1803.01271 (2018).
for music data
To compare with the original implementation, run
python ./music_forecasting_with_tcn.py --batch_size 32 --dropout 0.5 --epochs 2000 --ksize 3 --levels 2 --seq_len 100 --log_interval 2 --nhid 150 --lr 0.002 --grad_clip_value 0.4 --results_dir ./
python ./music_forecasting_with_tcn.py --dataset Nott --batch_size 32 --dropout 0.2 --epochs 2000 --ksize 6 --levels 4 --seq_len 250 --log_interval 2 --nhid 150 --lr 0.002 --grad_clip_value 0.4 --results_dir ./
"""
from examples.tcn.music import Music
from examples.tcn.temporal_convolutional_network import TCN
import argparse
import tensorflow as tf
import numpy as np

class TCNForMusic(TCN):
    def __init__(self, *args, **kwargs):
        super(TCNForMusic, self).__init__(*args, **kwargs)

    def run(self, sess, data_loader, log_interval=100, result_dir="./"):
        for i in range(num_iterations):

            X_data, y_data = next(data_loader)

            feed_dict = {self.input_placeholder: X_data, self.label_placeholder: y_data, self.training_mode: True}
            _, summary_train, total_loss_i, pred_i = sess.run([self.training_update_step, self.merged_summary_op_train, self.training_loss, self.sequence_output_same_features], feed_dict=feed_dict)

            self.summary_writer.add_summary(summary_train, i)

            if i % log_interval == 0:
                print("Step {}: Total: {}".format(i, total_loss_i))
                self.saver.save(sess, result_dir, global_step=i)

                feed_dict = {self.input_placeholder: data_loader.test[:, 0:seq_len, :], self.label_placeholder: data_loader.test[:, 1:, :], self.training_mode: False}
                val_loss, summary_val = sess.run([self.training_loss, self.merged_summary_op_val], feed_dict=feed_dict)

                self.summary_writer.add_summary(summary_val, i)

                print("Validation loss: {}".format(val_loss))

    def build_train_graph(self, lr, max_gradient_norm=None):
        with tf.variable_scope("input", reuse=True):
            self.input_placeholder = tf.placeholder(tf.float32, [None, self.max_len, self.n_features_in], name='input')
            self.label_placeholder = tf.placeholder(tf.float32, [None, self.max_len, self.n_features_in], name='input_shifted')

        self._build_network_graph(self.input_placeholder)
        self._get_predictions()

        with tf.variable_scope("training"):
            self.sequence_output_same_features = tf.layers.Dense(self.n_features_in, activation=tf.nn.sigmoid,kernel_initializer=tf.initializers.random_normal(0, 0.01), bias_initializer=tf.initializers.random_normal(0, 0.01))(self.prediction)

            eps = 1e-32
            self.training_loss = tf.reduce_mean(tf.reduce_sum(-1 * self.label_placeholder * tf.log(self.sequence_output_same_features + eps) - (1 - self.label_placeholder) * tf.log(1 - self.sequence_output_same_features + eps), axis=-1))

            summary_ops_train = []
            summary_ops_train.append(tf.summary.scalar("Training Loss", self.training_loss))
            self.merged_summary_op_train = tf.summary.merge(summary_ops_train)

            summary_ops_val = []
            summary_ops_val.append(tf.summary.scalar("Validation Loss", self.training_loss))
            self.merged_summary_op_val = tf.summary.merge(summary_ops_val)

            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.training_loss, params)

            if max_gradient_norm is not None:
                clipped_gradients = [tf.clip_by_norm(t, max_gradient_norm) for t in gradients]
            else:
                clipped_gradients = gradients

            grad_norm = tf.global_norm(clipped_gradients)
            summary_ops_train.append(tf.summary.scalar("Grad Norm", grad_norm))
            self.merged_summary_op_train = tf.summary.merge(summary_ops_train)


            # Optimization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            with tf.control_dependencies(update_ops):
                self.training_update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int,
                    help="Number of time points in each input sequence",
                    default=200)
parser.add_argument('--log_interval', type=int, default=1000, help="frequency, in number of iterations, after which loss is evaluated")
parser.add_argument('--results_dir', type=str, help="Directory to write results to", default='./')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--ksize', type=int, default=6,
                    help='kernel size (default: 6)')
parser.add_argument('--levels', type=int, default=7,
                    help='# of levels (default: 7)')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--nhid', type=int, default=27,
                    help='number of hidden units per layer (default: 27)')
parser.add_argument('--grad_clip_value', type=float, default=0.4,
                    help='value to clip each element of gradient')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--datadir', type=str, default="../data/",
                    help='dir to download data if not already present')
parser.add_argument('--dataset', default='JSB', choices=['JSB', 'Nott'],
                        help='type of data to use (JSB, Nott)')
parser.set_defaults()
args = parser.parse_args()
print(args)

hidden_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
seq_len = args.seq_len
batch_size = args.batch_size
n_epochs = args.epochs

music_dataset = Music(data_dir=args.datadir, seq_len=seq_len, dataset=args.dataset)
seq_len = music_dataset.seq_len
n_train = music_dataset.train.shape[0]
num_iterations = int(n_train * n_epochs * 1.0 / batch_size)
n_features = music_dataset.train.shape[2]

model = TCNForMusic(seq_len, n_features, hidden_sizes, kernel_size=kernel_size, dropout=dropout, last_timepoint=False)

model.build_train_graph(args.lr, max_gradient_norm=args.grad_clip_value)

model.set_up_callbacks(args.results_dir)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
model.run(sess, music_dataset, log_interval=args.log_interval, result_dir=args.results_dir)

