import os
import sys
import time

# Run image recovery in single digit mode for many lambdas
def inner_loop(mode, penalty_mode, num_steps, pgd, digits, lambdas, timestamp, dataset):
    folder = f"image_lambda_runs/{mode}/{penalty_mode.replace(' ','_')}/num-recovery-steps_{num_steps}/{pgd}/{timestamp}"
    for digit in digits:
        for i, lambd in enumerate(lambdas):
            cmd = f'python3 sparse_recovery_main.py --mode {mode} ' \
                    f'--dataset {dataset} ' \
                    f'--digit {digit} ' \
                    f'--recovery-penalty-mode "{penalty_mode}" ' \
                    f'--recovery-lambd {lambd} ' \
                    f'--run-suffix _lambda_{lambd} ' \
                    f'--run-dir {folder}/{digit}/{i:0>2d}' \
                    f' --recovery-num-steps {num_steps}' \
                    f' --{pgd} ' \
                    f' --discriminator-model-file {dmf} '
            print(cmd)
            out = os.system(cmd)
            if out != 0:
                print('Encountered error, exiting...')
                sys.exit(1)

# Run image recovery in dataset mode for many lambdas
def inner_loop_dataset(mode, penalty_mode, num_steps, pgd, digits, lambdas, timestamp, dataset, dmf, dataset_len,
        batch_size):
    folder = f"image_lambda_runs/{dataset}/{mode}/{penalty_mode.replace(' ','_')}/num-recovery-steps_{num_steps}/{pgd}/{timestamp}"
    for i, lambd in enumerate(lambdas):
        cmd = f'python3.6 sparse_recovery_main.py ' \
                f'--mode {mode} ' \
                f'--dataset {dataset} ' \
                f'--recovery-penalty-mode "{penalty_mode}" ' \
                f'--recovery-lambd {lambd} ' \
                f'--run-suffix _lambda_{lambd} ' \
                f'--run-dir {folder}/{i:0>2d}' \
                f' --recovery-num-steps {num_steps}' \
                f' --{pgd} ' \
                f' --discriminator-model-file {dmf} '\
                f' --dataset-len {dataset_len} ' \
                f' --recovery-batch-size {batch_size}'\
                f' --recovery-balance-classes '\

        print(cmd)
        out = os.system(cmd)
        if out != 0:
            print('Encountered error, exiting...')
            sys.exit(1)

def mega_loop():
    dataset = 'cifar'
    #mode = 'single-digit'
    mode = 'gen-dataset'
    dmf = 'train_runs/0033-May08_22-02-11_adv-train-fresh-full_cifar/ckpt/model_opt_sched/model_opt_sched_0199.pt'
    dataset_len = 100
    batch_size = 100
    digits = list(range(10))
    #digits = [0]
    #lambdas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    lambdas = [5.0, 10.0, 20.0, 50.0, 100.0]
    #lambdas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    penalty_mode = "input only"
    num_steps = 400
    timestamp = time.strftime('%b%d_%H-%M-%S')
    pgd = 'recovery-enable-pgd' # or 'enable-pgd'


    # for penalty_mode in ["input only", "all layers"]:
    for penalty_mode in ["input only", ]:
        for pgd in ['recovery-enable-pgd', ]:
        # for pgd in ['enable-pgd', 'disable-pgd']:
            if penalty_mode == 'input only' and pgd == 'disable-pgd':
                continue
            print(penalty_mode, pgd)
            inner_loop_dataset(mode, penalty_mode, num_steps, pgd, digits, lambdas, timestamp, dataset, dmf,
                    dataset_len, batch_size)

mega_loop()

