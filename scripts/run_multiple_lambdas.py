import os

for lambd in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    cmd = f'python3 mnist_sparse_recovery.py --lambd {lambd} --run-suffix _lambda_{lambd}'
    print(cmd)
    os.system(cmd)


