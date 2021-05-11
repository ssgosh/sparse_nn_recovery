import os
import time
from subprocess import Popen, DEVNULL

dataset = 'cifar'

if dataset == 'mnist':
    resume_dir = 'train_runs/0022-May08_11-23-42_adv-train-fresh-full_mnist'
    resume_epoch = 0
    epochs = 2
else:
    resume_dir = 'train_runs/0017-May08_00-34-02_full-cifar_cifar'
    resume_epoch = 49
    epochs = 51

for bs in  [ 2048, 32, 64, 128, 256, 512, 1024, ]:
    cmd = f'python3.9 run_named_expt.py --expt adv-train-fresh-full --dataset cifar --sparse-dataset ' \
            f'--resume-epoch {resume_epoch} ' \
            f'--resume-dir {resume_dir} ' \
            f'--num-adversarial-images-epoch-mode 2048 ' \
            f'--recovery-batch-size {bs} ' \
            f'--recovery-num-steps 100 --epochs {epochs} --early-epoch ' \
            f'--num-pretrain-epochs 0'
    #cmd = f'echo foobar {bs}'
    cmd_lst = cmd.split()
    time_start = time.perf_counter()
    #p = Popen(cmd_lst, stdout=DEVNULL, stderr=DEVNULL)
    p = Popen(cmd_lst)
    p.wait()
    time_end = time.perf_counter()
    print(f'bs = {bs}, time =', time_end - time_start, 'seconds')
    #os.system(cmd)
