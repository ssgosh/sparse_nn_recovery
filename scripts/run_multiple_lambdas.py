import os
import sys

for digit in range(10):
    for i, lambd in enumerate([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
        100.0]):
        cmd = f'python3 mnist_sparse_recovery.py --mode single-digit ' \
                f'--digit {digit} ' \
                f'--lambd {lambd} ' \
                f'--run-suffix _lambda_{lambd} ' \
                f'--run-dir image_lambda_runs/{digit}/{i:0>2d}' \
                f' --num-steps 1000'
        print(cmd)
        out = os.system(cmd)
        if out != 0:
            print('Encountered error, exiting...')
            sys.exit(1)


