import os

for i, lambd in enumerate([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
    100.0]):
    cmd = f'python3 mnist_sparse_recovery.py --mode single-digit ' \
            f'--digit 0 ' \
            f'--lambd {lambd} ' \
            f'--run-suffix _lambda_{lambd} ' \
            f'--run-dir image_lambda_runs/{i:0>2d}' \
            ' --num-steps 10'
    print(cmd)
    os.system(cmd)


