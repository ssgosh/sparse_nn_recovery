import argparse
import os
import signal
import subprocess
import sys
from subprocess import Popen, PIPE, STDOUT

from utils import runs_helper as rh
from utils.seed_mgr import SeedManager


def log_cmd(cmd, f):
    out = subprocess.run(['-c', cmd], shell=True, capture_output=True)
    f.write(out.stdout)


def save_git_info(diff_file):
    with open(diff_file, 'wb') as f:
        f.write(bytes("\n**************   Git Branch Information   **********:\n", 'utf-8'))
        cmd = 'git branch -vv'
        log_cmd(cmd, f)

        f.write(bytes("\n**************   Git Log Information   **********:\n", 'utf-8'))
        # cmd = 'git log --oneline --graph'.split()
        cmd = 'git log --oneline --graph | head -20'
        log_cmd(cmd, f)

        f.write(bytes("\n**************   Git Diff with HEAD   **********:\n", 'utf-8'))
        cmd = 'git diff HEAD'
        log_cmd(cmd, f)


class NamedExpt:
    """
    Manages Experiment Presets
    """
    def __init__(self):
        self.seed_mgr = SeedManager.get_project_seed_mgr()

        self.names = [
            'quick', 'quick-debug', # Only for checking if the pipeline works without any python errors. Doesn't care about algo output
            'quick-opt', # For quickly testing if the optimization is somewhat working, with minimal number of epochs, batches etc
            'full-sparse',  # For full expt, with data loaded in normal mode
            'full-non-sparse',  # For full expt, with data loaded in non-sparse mode (add constant pixel)
            'full-cifar', # For full cifar-10 expts, with epochs etc set for cifar dataset
            'pretrain-MNIST_B', # For pretraining on dataset B
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
        args.run_suffix = f"_{args.expt}_{args.dataset}"
        rh.setup_run_dir(args, 'train_runs')

        # Save git information in the run directory before proceeding
        save_git_info(f'{args.run_dir}/gitinfo.diff')

        python3 = os.environ['PYTHON3']
        #f'--run-suffix _{seed_hash} '
        cmd = f'{python3} mnist_train.py ' \
              f'--name {name} ' \
              f'--seed {seed} ' \
              f'--run-dir {args.run_dir} ' \
              f'--train-mode adversarial-epoch ' \
              f'--dataset {dataset} '
        if name in ['quick', 'quick-debug',]:
            cmd = cmd + \
                  f'--early-epoch ' \
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
        elif args.expt == 'full-sparse':
            cmd = cmd + \
                    f'--sparse-dataset ' 
        elif args.expt == 'full-non-sparse':
            cmd = cmd + \
                    f'--non-sparse-dataset ' 
        elif args.expt == 'full-cifar':
            assert args.dataset.lower() == 'cifar'
            cmd = cmd + \
                    f'--sparse-dataset ' \
                    f'--epochs 350 ' \
                    f'--num-pretrain-epochs 50 ' \
                    f'--recovery-batch-size 256 ' \
                    f'--num-adversarial-images-epoch-mode 1024 ' \
                    f'--recovery-num-steps 3500 ' \
                    f'--batch-size 128 ' \
                    f'--adv-loss-weight 0.1 ' \
                    f'--no-lambda-annealing ' \
                    f'--adv-data-generation-steps 10 ' \
                    f'--recovery-sparsity-threshold 600'
        elif args.expt == 'pretrain-MNIST_B':
            cmd = cmd + \
                    f'--dataset MNIST_B ' \
                    f'--epochs 15 ' \
                    f'--num-pretrain-epochs 14 '

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
                f.write(" ".join(cmd_lst) +"\n")
                f.flush()
                foo = self.run_command(cmd_lst)
                #print(foo)
                for l, rc in foo:
                    print(l, end="", flush=True)
                    f.write(l)
                    f.flush()
        except KeyboardInterrupt:
            # Following should work to record the traceback in the logfile but doesn't :-(
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

