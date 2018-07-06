from pathlib import Path

from train import run_aspect_sequence_tagging

# TODO: add click and params for cmd line
if __name__ == '__main__':
    datasets_path = Path('/home/lukasz/github/phd/sentiment-backend/aspects/data/aspects/bing_liu/bio_tags')
    conll_train_files = list(datasets_path.glob('*train.conll'))
    conll_test_files = list(datasets_path.glob('*test.conll'))

    embedding_model = '/home/lukasz/data/glove.840B.300d.txt'

    for train_file in conll_train_files:
        test_file = [f for f in conll_test_files if train_file.stem.replace('train', '') in f.as_posix()][0]
        run_aspect_sequence_tagging(
            train_file=train_file.as_posix(),
            test_file=test_file.as_posix(),
            embedding_model=embedding_model,
            tag_num=2,
            epoch=3,
        )
