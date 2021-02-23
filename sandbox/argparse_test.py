import argparse
import sys

parser1 = argparse.ArgumentParser(description='Not suppressing defaults',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser2 = argparse.ArgumentParser(description='Suppressing defaults',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

def add_args(parser, suppress):
    available_train_modes = ['normal', 'adversarial-continuous', 'adversarial-epoch', 'adversarial-batches']
    parser.add_argument('--train-mode', type=str, default='normal',
                        metavar='MODE',
                        choices=available_train_modes,
                        help='Training mode. One of: ' + ', '.join(available_train_modes))
    parser.add_argument('--pretrain', action='store_true', default=True,
                        dest='pretrain',
                        help='Pretrain before adversarial training')
    if suppress:
        for arg in parser._optionals._actions:
            print(arg)
            arg.default = argparse.SUPPRESS
            #arg.set_default(argparse.SUPPRESS)

add_args(parser1, False)
add_args(parser2, True)
str1 = list(sys.argv[1:])
str2 = list(sys.argv[1:])
args1 = parser1.parse_args(str1)
args2 = parser2.parse_args(str2)

print(vars(args1))
print(vars(args2))

