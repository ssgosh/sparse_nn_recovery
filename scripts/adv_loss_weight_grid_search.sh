#!/bin/bash

for weight in `echo 0.2 0.5 1.0`
do
    CMD="python3.9 run_named_expt.py --expt adv-train-fresh-full --dataset cifar --adv-loss-weight $weight"
    echo $CMD
    $CMD
done
