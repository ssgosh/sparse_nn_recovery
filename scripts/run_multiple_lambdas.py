import os
import sys
import time

def inner_loop(mode, penalty_mode, num_steps, pgd, digits, lambdas, timestamp):
    folder = f"image_lambda_runs/{mode}/{penalty_mode.replace(' ','_')}/num-recovery-steps_{num_steps}/{pgd}/{timestamp}"
    for digit in digits:
        for i, lambd in enumerate(lambdas):
            cmd = f'python3 mnist_sparse_recovery.py --mode {mode} ' \
                    f'--digit {digit} ' \
                    f'--penalty-mode "{penalty_mode}" ' \
                    f'--lambd {lambd} ' \
                    f'--run-suffix _lambda_{lambd} ' \
                    f'--run-dir {folder}/{digit}/{i:0>2d}' \
                    f' --recovery-num-steps {num_steps}' \
                    f' --{pgd}' \

            print(cmd)
            out = os.system(cmd)
            if out != 0:
                print('Encountered error, exiting...')
                sys.exit(1)


def mega_loop():
    mode = 'single-digit'
    digits = list(range(10))
    #digits = [0]
    #lambdas = [0.1, 0.2]
    lambdas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    penalty_mode = "input only"
    num_steps = 1000
    timestamp = time.strftime('%b%d_%H-%M-%S')
    pgd = 'enable-pgd' # or 'enable-pgd'


    for penalty_mode in ["input only", "all layers"]:
        for pgd in ['enable-pgd', 'disable-pgd']:
            if penalty_mode == 'input only' and pgd == 'disable-pgd':
                continue
            print(penalty_mode, pgd)
            inner_loop(mode, penalty_mode, num_steps, pgd, digits, lambdas, timestamp)

mega_loop()

