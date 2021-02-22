if __name__ == '__main__':
    import sys
    sys.path.append(".")

from utils.tensorboard_helper import TensorBoardHelper
from random import shuffle

def test_tbh_add_list():
    dirname = "test_runs_87432"
    tbh = TensorBoardHelper(dirname)
    lst = list(range(97))
    for i in range(10):
        tbh.add_list(lst, "test_tag/test_list", num_per_row=10, global_step=i)
        shuffle(lst)

if __name__ == '__main__':
    test_tbh_add_list()

