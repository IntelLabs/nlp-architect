"""
This script replicates the experiments in the following paper for the synthetic "adding" data:
Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling." arXiv preprint arXiv:1803.01271 (2018).

To compare with the original implementation, run
python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 6 --levels 7
--seq_len 200 --log_interval 100 --nhid 27 --lr 0.002 --results_dir ./

python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 7 --levels 7
--seq_len 400 --log_interval 100 --nhid 27 --lr 0.002 --results_dir ./

python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 8 --levels 8
--seq_len 600 --log_interval 100 --nhid 24 --lr 0.002 --results_dir ./
"""
import os
import argparse
import tensorflow as tf
from nlp_architect.models.temporal_convolutional_network import TCN
from examples.word_language_model_with_tcn.toy_data.adding import Adding
from nlp_architect.utils.io import validate, validate_existing_directory, \
    validate_existing_filepath, validate_parent_exists, check_size


class TCNForAdding(TCN):
    """
    Main class that defines training graph and defines training run method for the adding problem
    """
    def __init__(self, *args, **kwargs):
        super(TCNForAdding, self).__init__(*args, **kwargs)
        self.input_placeholder = None
        self.label_placeholder = None
        self.prediction = None
        self.training_loss = None
        self.merged_summary_op_train = None
        self.merged_summary_op_val = None
        self.training_update_step = None

    def run(self, data_loader, num_iterations=1000, log_interval=100, result_dir="./"):
        """
        Runs training
        Args:
            data_loader: iterator, Data loader for adding problem
            num_iterations: int, number of iterations to run
            log_interval: int, number of iterations after which to run validation and log
            result_dir: str, path to results directory

        Returns:
            None
        """
        summary_writer = tf.summary.FileWriter(os.path.join(result_dir, "tfboard"),
                                               tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=None)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        for i in range(num_iterations):

            x_data, y_data = next(data_loader)

            feed_dict = {self.input_placeholder: x_data, self.label_placeholder: y_data,
                         self.training_mode: True}
            _, summary_train, total_loss_i = sess.run([self.training_update_step,
                                                       self.merged_summary_op_train,
                                                       self.training_loss], feed_dict=feed_dict)

            summary_writer.add_summary(summary_train, i)

            if i % log_interval == 0:
                print("Step {}: Total: {}".format(i, total_loss_i))
                saver.save(sess, result_dir, global_step=i)

                feed_dict = {self.input_placeholder: data_loader.test[0],
                             self.label_placeholder: data_loader.test[1], self.training_mode: False}
                val_loss, summary_val = sess.run([self.training_loss, self.merged_summary_op_val],
                                                 feed_dict=feed_dict)

                summary_writer.add_summary(summary_val, i)

                print("Validation loss: {}".format(val_loss))

    def build_train_graph(self, lr, max_gradient_norm=None):
        """
        Method that builds the graph for training
        Args:
            lr: float, learning rate
            max_gradient_norm: float, maximum gradient norm value for clipping

        Returns:
            None
        """
        with tf.variable_scope("input", reuse=True):
            self.input_placeholder = tf.placeholder(tf.float32,
                                                    [None, self.max_len, self.n_features_in],
                                                    name='input')
            self.label_placeholder = tf.placeholder(tf.float32, [None, 1], name='labels')

        self.prediction = self.build_network_graph(self.input_placeholder, last_timepoint=True)

        with tf.variable_scope("training"):
            self.training_loss = tf.losses.mean_squared_error(self.label_placeholder,
                                                              self.prediction)

            summary_ops_train = [tf.summary.scalar("Training Loss", self.training_loss)]
            self.merged_summary_op_train = tf.summary.merge(summary_ops_train)

            summary_ops_val = [tf.summary.scalar("Validation Loss", self.training_loss)]
            self.merged_summary_op_val = tf.summary.merge(summary_ops_val)

            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.training_loss, params)
            if max_gradient_norm is not None:
                clipped_gradients = [tf.clip_by_norm(t, max_gradient_norm) for t in gradients]
            else:
                clipped_gradients = gradients

            # Optimization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(lr)
            with tf.control_dependencies(update_ops):
                self.training_update_step = optimizer.apply_gradients(zip(clipped_gradients,
                                                                          params))


def main(args):
    """
    Main function
    Args:
        args: output of argparse with all input arguments

    Returns:
        None
    """
    n_features = 2
    hidden_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    seq_len = args.seq_len
    n_train = 50000
    n_val = 1000
    batch_size = args.batch_size
    n_epochs = args.epochs
    num_iterations = int(n_train * n_epochs * 1.0 / batch_size)
    results_dir = os.path.abspath(args.results_dir)

    adding_dataset = Adding(seq_len=seq_len, n_train=n_train, n_test=n_val)

    with tf.device('/gpu:0'):
        model = TCNForAdding(seq_len, n_features, hidden_sizes, kernel_size=kernel_size,
                             dropout=dropout)

        model.build_train_graph(args.lr, max_gradient_norm=args.grad_clip_value)

        model.run(adding_dataset, num_iterations=num_iterations, log_interval=args.log_interval,
                  result_dir=results_dir)


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--seq_len', type=int,
                    help="Number of time points in each input sequence",
                    default=200)
PARSER.add_argument('--log_interval', type=int, default=100,
                    help="frequency, in number of iterations, after which loss is evaluated")
PARSER.add_argument('--results_dir', type=validate_parent_exists,
                    help="Directory to write results to", default='./')
PARSER.add_argument('--dropout', type=float, default=0.0, action=check_size(0, 1),
                    help='dropout applied to layers, between 0 and 1 (default: 0.0)')
PARSER.add_argument('--ksize', type=int, default=6,
                    help='kernel size (default: 6)')
PARSER.add_argument('--levels', type=int, default=7,
                    help='# of levels (default: 7)')
PARSER.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
PARSER.add_argument('--nhid', type=int, default=27,
                    help='number of hidden units per layer (default: 27)')
PARSER.add_argument('--grad_clip_value', type=float, default=None,
                    help='value to clip each element of gradient')
PARSER.add_argument('--batch_size', type=int, default=32,
                    help='Batch size')
PARSER.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs')
PARSER.set_defaults()
ARGS = PARSER.parse_args()
main(ARGS)
