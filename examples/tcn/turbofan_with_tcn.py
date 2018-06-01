""""
To run this script, use the command:
python ./turbofan_with_tcn.py --batch_size 128 --dropout 0.1 --ksize 4 --levels 4 --seq_len 50 --log_interval 100 --nhid 50 --lr 0.002 --grad_clip_value 0.4 --save_plots --epochs 200 --results_dir ./ -b gpu
"""
from topologies.temporal_convolutional_network import tcn
from ngraph.frontends.neon import ArrayIterator
import ngraph as ng
from ngraph.frontends.neon import Adam, Layer, Affine, Identity, GaussianInit, Sequential, Rectlin
from training.timeseries_trainer import TimeseriesTrainer
from datasets.turbofan import TurboFan
import os
from utils.arguments import default_argparser

parser = default_argparser()
parser.add_argument('--skip', default=1, type=int, help='skip length for sliding window')
parser.add_argument('--datadir', type=str, default="../data/",
                    help='dir to download data if not already present')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--grad_clip_value', type=float, default=None,
                    help='value to clip each element of gradient')
parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard',
                    help='directory to save tensorboard summary to')
args = parser.parse_args()

if args.save_plots:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        args.save_plots = False

hidden_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = 1 - args.dropout # amount to keep
seq_len = args.seq_len
batch_size = args.batch_size
n_epochs = args.epochs

receptive_field_last_t = (kernel_size - 1) * (2 ** args.levels - 1)  # receptive field of last time-point

if seq_len - receptive_field_last_t > 5:
    print("WARNING: Given these parameters, the last time-point's receptive field does not cover the entire sequence length")
    print("Difference in coverage = %d time-points" % (seq_len - receptive_field_last_t))

turbofan_dataset = TurboFan(data_dir=args.datadir, T=seq_len, skip=args.skip, max_rul_predictable=130)
train_samples = len(turbofan_dataset.train['X']['data'])
num_iterations = (n_epochs * train_samples) // batch_size
n_features = turbofan_dataset.train['X']['data'].shape[2]
n_output_features = 1

train_iterator = ArrayIterator(turbofan_dataset.train, batch_size, total_iterations=num_iterations, shuffle=True)
test_iterator = ArrayIterator(turbofan_dataset.test, batch_size)
train_set_one_epoch = ArrayIterator(turbofan_dataset.train, batch_size, shuffle=False)

# Name and create axes
batch_axis = ng.make_axis(length=batch_size, name="N")
time_axis = ng.make_axis(length=seq_len, name="REC")
feature_axis = ng.make_axis(length=n_features, name="F")
out_axis = ng.make_axis(length=n_output_features, name="Fo")

in_axes = ng.make_axes([batch_axis, time_axis, feature_axis])
out_axes = ng.make_axes([batch_axis, out_axis])

# Build placeholders for the created axes
inputs = dict(X=ng.placeholder(in_axes), y=ng.placeholder(out_axes),
              iteration=ng.placeholder(axes=()))

# take only the last timepoint of output sequence to predict RUL
last_timepoint = [lambda op: ng.tensor_slice(op, [slice(seq_len-1, seq_len, 1) if ax.name == "W" else slice(None) for ax in op.axes])]
affine_layer = Affine(axes=out_axis, weight_init=GaussianInit(0, 0.01), activation=Rectlin())
model = Sequential([lambda op: ng.map_roles(op, {'F': 'C', 'REC': 'W'})] + tcn(n_features, hidden_sizes, kernel_size=kernel_size, dropout=dropout).layers + last_timepoint + [affine_layer])


# Optimizer
optimizer = Adam(learning_rate=args.lr, gradient_clip_value=args.grad_clip_value)

# Define the loss function (categorical cross entropy, since each musical key on the piano is encoded as a binary value)
fwd_prop = model(inputs['X'])
fwd_prop = ng.axes_with_order(fwd_prop, out_axes)
train_loss = ng.squared_L2(fwd_prop - inputs['y'])
with Layer.inference_mode_on():
    preds = model(inputs['X'])
    preds = ng.axes_with_order(preds, out_axes)
eval_loss = ng.mean(ng.squared_L2(preds - inputs['y']), out_axes=())
eval_computation = ng.computation([eval_loss], "all")
predict_computation = ng.computation([preds], "all")


# Cost calculation
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation(batch_cost, "all")

trainer = TimeseriesTrainer(optimizer, train_computation, eval_computation, predict_computation, inputs,
                            model_graph=[model], tensorboard_dir=args.tensorboard_dir)
trainer.summary()

out_folder = os.path.join(args.results_dir, "results-turbofan-{}-batch_size-{}-dropout-{}-ksize-{}-levels-{}-seq_len-{}-nhid".format(batch_size, args.dropout, kernel_size, args.levels, seq_len, args.nhid))
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
trainer.train(train_iterator, test_iterator, n_epochs=args.epochs, log_interval=args.log_interval, save_plots=args.save_plots, results_dir=out_folder)


if args.save_plots:
    # Compute the predictions on the training and test sets for visualization
    train_preds = trainer.predict(train_set_one_epoch)
    train_target = turbofan_dataset.train['y']['data']

    test_preds = trainer.predict(test_iterator)
    test_target = turbofan_dataset.test['y']['data']

    # Visualize the model's predictions on the training and test sets
    plt.figure()
    plt.scatter(train_preds[:, 0], train_target[:, 0])
    plt.xlabel('Training Predictions')
    plt.ylabel('Training Targets')
    plt.title('Predictions on training set')
    plt.savefig(os.path.join(out_folder, 'preds_training_output.png'))

    plt.figure()
    plt.scatter(test_preds[:, 0], test_target[:, 0])
    plt.xlabel('Validation Predictions')
    plt.ylabel('Validation Targets')
    plt.title('Predictions on validation set')
    plt.savefig(os.path.join(out_folder, 'preds_validation_output.png'))
