import argparse
import os
import dataset


def run(path):
    audio_files = os.listdir(path)[:100]

    labels, features = dataset.load(audio_files, path)

    print(features.shape)
    print(labels.shape)


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
