import argparse
import os
import sys
from subprocess import Popen, PIPE, STDOUT

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
        seed, seed_hash = self.seed_mgr.get_random_seed_hashid()
        name = args.expt
        dataset = args.dataset
        cmd = 'python3 mnist_train.py ' \
              f'--name {name} ' \
              f'--seed {seed} ' \
              f'--run-suffix _{seed_hash} ' \
              f'--dataset {dataset} '
        if name in ['quick', 'quick-debug',]:
            cmd = cmd + \
                  f'--early-epoch ' \
                  f'--train-mode adversarial-epoch ' \
                  f'--adversarial-classification-mode max-entropy ' \
                  f'--epochs 4 ' \
                  f'--recovery-num-steps 1 ' \
                  f'--num-adversarial-images-epoch-mode 128 ' \
                  f'--recovery-batch-size 128 ' \
                  f'--num-batches-early-epoch 10 '
        elif args.expt == 'quick-opt':
            cmd = cmd + \
                  f'--early-epoch ' \
                  f'--train-mode adversarial-epoch ' \
                  f'--adversarial-classification-mode max-entropy ' \
                  f'--epochs 6 ' \
                  f'--recovery-num-steps 100 ' \
                  f'--num-adversarial-images-epoch-mode 128 ' \
                  f'--recovery-batch-size 128 ' \
                  f'--num-batches-early-epoch 100 '
        elif args.expt == 'full':
            pass

        # Overrides anything specified in this script via the command-line
        cmd_lst = cmd.split() + extra_args

        print(" ".join(cmd_lst))

        # Remove stupid python buffering
        os.environ["PYTHONUNBUFFERED"] = "1"
        with Popen(cmd_lst, stdout=PIPE, stderr=STDOUT, bufsize=1, text=True) as p, \
                open(f'logs/logfile_{seed_hash}.txt', 'a') as file:
            for line in p.stdout:
                sys.stdout.write(line)
                file.write(line)

if __name__ == '__main__':
    expt = NamedExpt()
    expt.main()