import argparse
import random
import sys

from utils.seed_mgr import SeedManager


class NamedExpt:
    def __init__(self):
        self.names = [
            'quick', 'quick-debug', # Only for checking if the pipeline works without any python errors. Doesn't care about algo output
            'quick-opt', # For quickly testing if the optimization is somewhat working, with minimal number of epochs, batches etc
            'full',  # For full expt
        ]
        self.parser = argparse.ArgumentParser(description='Named Experiments',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--expt', type=str, metavar='MODE', choices=self.names, required=True,
                            help='Training mode. One of: ' + ', '.join(self.names))
        self.seed_mgr = SeedManager.get_project_seed_mgr()

    def main(self):
        args = self.parser.parse_args()

        #seed_id = self.seed_mgr.get_random_seed_hashid()
        seed = self.seed_mgr.get_random_seed()
        if args.expt == 'quick-debug' or args.expt == 'quick':
            cmd = 'python3 mnist_train.py ' \
                  f'--seed {seed} ' \
                  '--dataset MNIST_A ' \
                  '--early-epoch ' \
                  '--train-mode adversarial-epoch ' \
                  '--adversarial-classification-mode max-entropy ' \
                  '--epochs 4 ' \
                  '--recovery-num-steps 1 ' \
                  '--num-adversarial-images-epoch-mode 128 ' \
                  '--recovery-batch-size 128 ' \
                  '--num-batches-early-epoch 10'
        elif args.expt == 'quick-opt':
            pass
        elif args.expt == 'full':
            pass

