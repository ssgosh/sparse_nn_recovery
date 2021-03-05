import argparse
import os
import random
import sys

from utils.seed_mgr import SeedManager


class NamedExpt:
    """
    Manages Experiment Presets
    """
    def __init__(self):
        self.seed_mgr = SeedManager.get_project_seed_mgr()

        self.names = [
            'quick', 'quick-debug', # Only for checking if the pipeline works without any python errors. Doesn't care about algo output
            'quick-opt', # For quickly testing if the optimization is somewhat working, with minimal number of epochs, batches etc
            'full',  # For full expt
        ]
        self.parser = argparse.ArgumentParser(description='Named Experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--expt', type=str, metavar='MODE', choices=self.names, required=True, help='One of: ' + ', '.join(self.names))
        self.parser.add_argument('--dataset', type=str, metavar='', default='MNIST_A')

    def main(self):
        #args = self.parser.parse_args()
        args, extra_args = self.parser.parse_known_args()
        print(args)
        print(extra_args)

        #seed_id = self.seed_mgr.get_random_seed_hashid()
        seed = self.seed_mgr.get_random_seed()
        name = args.expt
        dataset = args.dataset
        if name in ['quick', 'quick-debug',]:
            cmd = 'python3 mnist_train.py ' \
                  f'--seed {seed} ' \
                  f'--dataset {dataset} ' \
                  f'--early-epoch ' \
                  f'--train-mode adversarial-epoch ' \
                  f'--adversarial-classification-mode max-entropy ' \
                  f'--epochs 4 ' \
                  f'--recovery-num-steps 1 ' \
                  f'--num-adversarial-images-epoch-mode 128 ' \
                  f'--recovery-batch-size 128 ' \
                  f'--num-batches-early-epoch 10 '
            cmd = cmd + " ".join(extra_args)
            print(cmd)
            #os.system(cmd)
        elif args.expt == 'quick-opt':
            pass
        elif args.expt == 'full':
            pass


if __name__ == '__main__':
    expt = NamedExpt()
    expt.main()