import sys
sys.path.append(".")
import matplotlib
import matplotlib.pyplot as plot
from matplotlib.pyplot import imshow

from datasets.dataset_helper_factory import DatasetHelperFactory

matplotlib.use('TkAgg')

dh1 = DatasetHelperFactory.get_new('mnist')
dh2 = DatasetHelperFactory.get_new('mnist_non_sparse')

print(dh1.non_sparse)
print(dh2.non_sparse)

print(dh1)
print(dh2)

d1 = dh1.get_dataset(which='train', transform='test')
d2 = dh2.get_dataset(which='train', transform='test')
for i in range(4):
    image, label = d1[i]
    imshow(image[0], cmap='gray', vmin=0., vmax=1.)
    plot.show()

    image, label = d2[i]
    imshow(image[0], cmap='gray', vmin=0., vmax=1.)
    plot.show()
