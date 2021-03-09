import argparse
import sys

from torchvision.transforms import transforms

from datasets.dataset_helper_factory import DatasetHelperFactory

sys.path.append(".")

parser = argparse.ArgumentParser(
    description='Splits train dataset into multiple train and validation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--dataset', type=str, default='mnist',
                    metavar='D',
                    help='Which dataset to split (e.g. MNIST)')
config = parser.parse_args()
transform = transforms.ToTensor()
dname = config.dataset
ds = DatasetHelperFactory.get(dname).get_dataset(which='train', transform=transform)

new_dname = f"{dname}_non_sparse"