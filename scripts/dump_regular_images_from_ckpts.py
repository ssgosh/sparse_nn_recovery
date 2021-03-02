import pathlib
import sys

import torch
import torchvision

from utils.ckpt_saver import CkptSaver
from datasets.dataset_helper_factory import DatasetHelperFactory
from utils.tensorboard_helper import TensorBoardHelper


class ImageDumper:
    def __init__(self):
        if len(sys.argv) != 2:
            print("Please provided log dir")
            sys.exit(1)
        self.logdir = pathlib.Path(sys.argv[1])
        self.ckpt_saver = CkptSaver(self.logdir)
        self.dataset_helper = DatasetHelperFactory.get('mnist')
        self.num_classes = self.dataset_helper.get_num_classes()
        self.tbh = TensorBoardHelper(self.logdir)
        self.device = torch.device('cpu')

    def get_regular_batch(self, images, targets, num_classes, num_per_class):
        entries = []
        for cls in range(num_classes):
            count = 0
            i = 0
            while count < num_per_class:
                if targets[i].item() == cls:
                    entries.append(images[i])
                    count += 1
                i += 1
        return torch.stack(entries)


    def iterate_over_epochs_get_images(self):
        image_batches_list = []
        try:
            epoch = 0
            while True:
                images, targets = self.ckpt_saver.load_images(epoch, self.device)
                images_batch = self.get_regular_batch(images, targets, self.num_classes, 10)
                image_batches_list.append(images_batch)
                epoch += 1
        except:
            print("Finished")

        return image_batches_list

    def dump_one_batch(self, images, filename, num_per_row):
        rng = self.dataset_helper.get_transformed_zero_one()
        img_grid = torchvision.utils.make_grid(
            images, num_per_row, normalize=True, range=rng, padding=2, pad_value=1.0, scale_each=True
        )
        # Resize to make it zoomed
        size = list(img_grid.shape)[1:]
        print(size)
        size = [ 3 * item for item in size]
        resize = torchvision.transforms.Resize(size)
        img_grid = resize(img_grid)
        torchvision.utils.save_image(img_grid, filename)

    def dump_image_batchess(self, image_batches_list, num_per_row):
        for epoch, images in enumerate(image_batches_list):
            filename = f"epoch_{epoch:0>2d}.jpg"
            self.dump_one_batch(images, filename, num_per_row)

    def main(self):
        images_list = self.iterate_over_epochs_get_images()
        self.dump_image_batchess(images_list, 10)

if __name__ == "__main__":
    img_dumper = ImageDumper()
    img_dumper.main()