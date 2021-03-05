import argparse
import random
import sys

from utils.seed_mgr import SeedManager


class NamedExpt:
    def __init__(self):
        self.names = [
            'quick-debug', # Only for checking if the pipeline works without any python errors. Doesn't care about algo output
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

        seed_id = self.seed_mgr.get_random_seed_hashid()
        if args.expt == 'quick-debug':
            pass
        elif args.expt == 'quick-opt':
            pass
        elif args.expt == 'full':
            pass

