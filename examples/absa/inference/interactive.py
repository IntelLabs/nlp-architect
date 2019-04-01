import argparse

from nlp_architect.models.absa.inference.inference import SentimentInference
from nlp_architect.utils.io import validate_existing_path


def main() -> None:
    parser = argparse.ArgumentParser(description='ABSA Inference')
    parser.add_argument('--aspects', type=validate_existing_path,
                        help='Path to aspect lexicon (csv)', required=True)
    parser.add_argument('--opinions', type=validate_existing_path, required=True,
                        help='Path to opinion lexicon (csv)')
    args = parser.parse_args()

    inference = SentimentInference(aspect_lex=args.aspects, opinion_lex=args.opinions)

    while True:
        doc = input('\nEnter sentence >> ')
        print(inference.run(doc))


if __name__ == '__main__':
    main()
