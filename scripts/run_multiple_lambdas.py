import os
import sys
import time

mode = 'single-digit'
digits = list(range(10))
#digits = [0]
#lambdas = [0.1, 0.2]
lambdas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
penalty_mode = "input only"
num_steps = 10
timestamp = time.strftime('%b%d_%H-%M-%S')
pgd = 'disable-pgd' # or 'enable-pgd'
folder = f"image_lambda_runs/{mode}/{penalty_mode.replace(' ','_')}/num-steps_{num_steps}/{pgd}/{timestamp}"
for digit in digits:
    for i, lambd in enumerate(lambdas):
        cmd = f'python3 mnist_sparse_recovery.py --mode {mode} ' \
                f'--digit {digit} ' \
                f'--penalty-mode "{penalty_mode}" ' \
                f'--lambd {lambd} ' \
                f'--run-suffix _lambda_{lambd} ' \
                f'--run-dir {folder}/{digit}/{i:0>2d}' \
                f' --num-steps {num_steps}' \
                f' --{pgd}' \

        print(cmd)
        out = os.system(cmd)
        if out != 0:
            print('Encountered error, exiting...')
            sys.exit(1)


