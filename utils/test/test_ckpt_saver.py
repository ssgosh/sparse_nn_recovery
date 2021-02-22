if __name__ == '__main__':
    import sys
    sys.path.append('.')

import torch
from utils.ckpt_saver import CkptSaver
def test_ckpt_save():
    ckpt = CkptSaver("ckpt_test/ckpt")
    images_list = [ torch.randn(20, 28, 28) for i in range(10) ]
    targets_list = [ torch.randint(high=10, low=0, size=(20,)) for i in range(10) ]
    for dataset_epoch, (images, targets) in enumerate(zip(images_list, targets_list)):
        ckpt.save_images(images, targets, dataset_epoch)

    for dataset_epoch in range(10):
        images, targets = ckpt.load_images(dataset_epoch)
        assert torch.all(images_list[dataset_epoch] == images).item()
        assert torch.all(targets_list[dataset_epoch] == targets).item()

if __name__ == '__main__':
    test_ckpt_save()

