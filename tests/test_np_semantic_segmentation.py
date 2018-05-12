import os
from neon.backends import gen_backend
from examples.np_semantic_segmentation.data import NpSemanticSegData, read_csv_file_data
from examples.np_semantic_segmentation.train import train_mlp_classifier
from examples.np_semantic_segmentation.inference import classify_collocation, extract_y_labels,\
    print_evaluation, write_results


def test_model_training():
    """
    Test model end2end training
    """
    data_path = os.path.abspath('fixtures/data/np_semantic_segmentation_prepared_data.csv')
    model_path = os.path.abspath('fixtures/data/np_semantic_segmentation')
    num_epochs = 200
    be = gen_backend(batch_size=64)
    # load data sets from file
    data_set = NpSemanticSegData(data_path, train_to_test_ratio=0.8)
    # train the mlp classifier
    train_mlp_classifier(data_set, model_path, num_epochs, {})
    assert os.path.isfile(os.path.abspath("fixtures/data/np_semantic_segmentation.prm")) is True


def test_model_inference():
    """
    Test model end2end inference
    """
    data_path = os.path.abspath('fixtures/data/np_semantic_segmentation_prepared_data.csv')
    output_path = os.path.abspath('fixtures/data/np_semantic_segmentation_output.csv')
    model_path = os.path.abspath('fixtures/data/np_semantic_segmentation.prm')
    num_epochs = 200
    callback_args = {}
    be = gen_backend(batch_size=10)
    print_stats = False
    data_set = NpSemanticSegData(data_path, train_to_test_ratio=1)
    results = classify_collocation(data_set, model_path, num_epochs, callback_args)
    if print_stats and (data_set.is_y_labels is not None):
        y_labels = extract_y_labels(data_path)
        print_evaluation(y_labels, results.argmax(1))
    write_results(results.argmax(1), output_path)
    assert os.path.isfile(os.path.abspath("fixtures/data/np_semantic_segmentation_output.csv"))\
           is True
    input_reader_list = read_csv_file_data(data_path)
    output_reader_list = read_csv_file_data(output_path)
    assert len(output_reader_list) == len(input_reader_list) - 1
    os.remove(model_path)
    os.remove(output_path)

