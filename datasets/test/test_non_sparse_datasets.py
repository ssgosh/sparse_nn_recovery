import sys
sys.path.append(".")

import matplotlib
import matplotlib.pyplot as plot

from datasets.dataset_helper_factory import DatasetHelperFactory


def show(msg, d1, d2):
    print(msg)
    for i in range(4):
        fig, (ax1, ax2) = plot.subplots(ncols=2)
        image, label = d1[i]
        ax1.imshow(image[0], cmap='gray', vmin=0., vmax=1.)

        image, label = d2[i]
        ax2.imshow(image[0], cmap='gray', vmin=0., vmax=1.)
        plot.show()


matplotlib.use('TkAgg')

dh1 = DatasetHelperFactory.get_new('mnist')
dh2 = DatasetHelperFactory.get_new('mnist_non_sparse')

print(dh1.non_sparse)
print(dh2.non_sparse)

#print(dh1)
#print(dh2)

d1_train = dh1.get_dataset(which='train', transform='test')
d2_train = dh2.get_dataset(which='train', transform='test')
show("Train datasets with test transform", d1_train, d2_train)

d1_train = dh1.get_dataset(which='train', transform='train')
d2_train = dh2.get_dataset(which='train', transform='train')
show("Train datasets with train transform", d1_train, d2_train)

d1_test = dh1.get_dataset(which='test', transform='test')
d2_test = dh2.get_dataset(which='test', transform='test')
show("Test datasets with Test transform", d1_test, d2_test)

d1_test = dh1.get_dataset(which='test', transform='train')
d2_test = dh2.get_dataset(which='test', transform='train')
show("Train datasets with train transform", d1_test, d2_test)
