import argparse
import os
import dataset
from sklearn.model_selection import ShuffleSplit
from collections import Counter

RANDOM_STATE = 42


def run(path):
    audio_files = os.listdir(path)

    labels, features = dataset.load(audio_files, path)

    shuffler = ShuffleSplit(n_splits=10, test_size=0.3, random_state=RANDOM_STATE)

    print(features.shape)
    print(labels.shape)

    for i, (train_index, test_index) in enumerate(shuffler.split(features)):
        print(f"Fold {i}:")

        features_train = features[train_index]
        labels_train = labels[train_index]
        features_test = features[test_index]
        labels_test = labels[test_index]

        counter_train = Counter(labels_train)
        counter_test = Counter(labels_test)

        print(f"  Train: index={features_train.shape, labels_train.shape}")
        print(f"  Train: counter={counter_train}")
        print(f"  Test:  index={features_test.shape, labels_test.shape}")
        print(f"  Test: counter={counter_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="./.data/recordings",
        type=str,
        help="Path to the downloaded data files",
    )
    args = parser.parse_args()

    run(args.path)
