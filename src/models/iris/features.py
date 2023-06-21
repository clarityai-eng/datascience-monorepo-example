import argparse

from config import cli
from feature_store.extract import extract_sklearn_feature_view


def main():
    parser = argparse.ArgumentParser()
    cli.add_model(parser, extract_sklearn_feature_view.Config)
    args = parser.parse_args()
    args_dict = {**vars(args), "sklearn_dataset": "iris"}
    config = extract_sklearn_feature_view.Config(**args_dict)
    task = extract_sklearn_feature_view.ExtractTask(config)
    task.run()


if __name__ == "__main__":
    main()
