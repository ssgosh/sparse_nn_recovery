import sys
sys.path.append('.')

import utils.plotter as plotter

import torch

def test_plot_single_digit():
    digit = 3
    image = torch.randn(28, 28)
    label = "Hello"
    plotter.plot_single_digit(image, digit, label, True)
    image = torch.randn(28, 28)
    plotter.plot_single_digit(image, digit, label, False)

test_plot_single_digit()
