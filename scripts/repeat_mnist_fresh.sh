#!/bin/bash

dataset='MNIST_A'
for i in `seq 1 5`
do
    CMD="python3.9 run_named_expt.py --expt adv-train-fresh-full --dataset $dataset"
    echo $CMD
    $CMD
done

