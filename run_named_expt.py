import argparse
import os
import signal
import sys
from subprocess import Popen, PIPE, STDOUT

from utils.seed_mgr import SeedManager
from utils import runs_helper as rh


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

        #print(args)
        #print(extra_args)

        #seed_id = self.seed_mgr.get_random_seed_hashid()
        seed, seed_hash = self.seed_mgr.get_random_seed_hashid()
        name = args.expt
        dataset = args.dataset

        # Create runs dir here, so that we can write to <run-dir>/logfile.txt
        args.run_dir = None
        args.run_suffix = f"_{args.expt}"
        rh.setup_run_dir(args, 'train_runs')

        #f'--run-suffix _{seed_hash} '
        cmd = 'python3 mnist_train.py ' \
              f'--name {name} ' \
              f'--seed {seed} ' \
              f'--run-dir {args.run_dir} ' \
              f'--dataset {dataset} '
        if name in ['quick', 'quick-debug',]:
            cmd = cmd + \
                  f'--early-epoch ' \
                  f'--train-mode adversarial-epoch ' \
                  f'--adversarial-classification-mode max-entropy ' \
                  f'--epochs 4 ' \
                  f'--recovery-num-steps 1 ' \
                  f'--num-adversarial-images-epoch-mode 32 ' \
                  f'--recovery-batch-size 32 ' \
                  f'--num-batches-early-epoch 1 '
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
        # From https://stackoverflow.com/a/52851238/2109112
        os.environ["PYTHONUNBUFFERED"] = "1" #text=True

        # From https://stackoverflow.com/a/34604684/2109112
        # Still has the '\r' problem
        # with Popen(cmd_lst, stdout=PIPE, stderr=STDOUT, bufsize=1, text=True) as p, \
        #         open(f'{args.run_dir}/logfile.txt', 'ab') as file:
        #     for line in p.stdout:
        #         sys.stdout.write(line)
        #         file.write(line)

        try:
            with open(f'{args.run_dir}/logfile.txt', 'a') as f:
                foo = self.run_command(cmd_lst)
                print(foo)
                for l, rc in foo:
                    print(l, end="", flush=True)
                    f.write(l)
                    f.flush()
        except KeyboardInterrupt:
            self.p.send_signal(signal.SIGINT)
            for l, rc in foo:
                print(l, end="", flush=True)
                f.write(l)
                f.flush()
            self.p.wait()
            raise


    # Plays nicely with '\r' and doesn't have any buffering issues.
    # Thanks to https://koldfront.dk/making_subprocesspopen_in_python_3_play_nice_with_elaborate_output_1594
    def run_command(self, cmd_lst):
        self.p = Popen(cmd_lst,
                  stdout=PIPE,
                  stderr=STDOUT,
                  universal_newlines=False)  # \r goes through

        nice_stdout = open(os.dup(self.p.stdout.fileno()), newline='')  # re-open to get \r recognized as new line
        for line in nice_stdout:
            yield line, self.p.poll()

        yield "", self.p.wait()



if __name__ == '__main__':
    expt = NamedExpt()
    expt.main()