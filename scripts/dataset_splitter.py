import argparse

from utils.dataset_helper import DatasetHelper

parser = argparse.ArgumentParser(
    description='Splits train dataset into multiple train and validation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dataset', type=str, default='normal',
                    metavar='D',
                    help='Which dataset to split (e.g. MNIST)')
config = parser.parse_args()
