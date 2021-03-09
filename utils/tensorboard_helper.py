import sys

import jsonpickle
import torch
import torchvision
import pandas as pd
import json

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

import utils.mnist_helper as mh
from datasets.dataset_helper_factory import DatasetHelperFactory


class TensorBoardHelper:

    def __init__(self, name=None):
        self.name = name
        self.writer = SummaryWriter(name)

        # Reset SummaryWriter after these many global_steps
        self.reset_steps = 5000
        self.next_reset = self.reset_steps

        self.image_zero, self.image_one = DatasetHelperFactory.get().get_transformed_zero_one()

    def close(self):
        print("Closing SummaryWriter")
        self.writer.close()

    def flush(self):
        print("Flushing SummaryWriter")
        self.writer.flush()

    def reset(self):
        self.close()
        print('Creating new SummaryWriter')
        self.writer = SummaryWriter(self.name)

    def reset_if_needed(self, global_step):
        if global_step >= self.next_reset:
            self.reset()
            self.next_reset = global_step + self.reset_steps

    def add_image_grid(self, images, tag, filtered, num_per_row, global_step):
        self.reset_if_needed(global_step)
        images = images.detach()
        rng = (self.image_zero, self.image_one) if filtered else None
        img_grid = torchvision.utils.make_grid(images, num_per_row, normalize=True,
                range=rng, padding=2, pad_value=1.0,
                scale_each=True)
        # Resize image grid
        #print(images.shape)
        #print(img_grid.shape)
        #size = img_grid.shape[1:]
        #print(size)
        #size = [3 * item for item in size]
        #print(size)
        img_grid = torch.unsqueeze(img_grid, 0)
        #print(img_grid.shape)
        img_grid = F.interpolate(img_grid, scale_factor=3.0).squeeze(0)
        #print(img_grid.shape)
        #sys.exit(1)
        #resize = torchvision.transforms.Resize(size)
        #toPIL = ToPILImage()
        #img_grid = resize(toPIL(   img_grid.detach().clone().cpu()   ))
        self.writer.add_image(tag, img_grid, global_step=global_step)


    # Plot 10 images in a 4x3 grid. One subplot per digit.
    def plot_image_batch(self, images, targets, label):
        num_images = len(images)
        nrows = 4
        ncols = 3
        plot.rcParams.update({'font.size' : 3 })
        fig, axes = plot.subplots(nrows=nrows, ncols=ncols, figsize=(1, 1))
        for i, ax in enumerate(axes.flat):
            if i >= num_images:
                fig.delaxes(ax)
                continue
            image = images[i][0]
            title = "%d : %s" % (targets[i], label)
            #plot_image_on_axis(ax, image, title, fig, vmin=-0.5, vmax=2.0)
            plot_image_on_axis(ax, image, title, fig)#, vmin=-0.5, vmax=2.0)

        #plot.tight_layout(pad=2.)
        #plot.savefig(filename)
        #plot.clf()
        #plot.rcParams.update({'font.size' : 10 })
        # Close this or we're gonna have a bad time with OOM if
        # called from within ipython
        #plot.close() 
        return fig

    # XXX: Don't use on every batch since this is too slow
    def add_figure(self, images, tag, global_step, label):
        fig = self.plot_image_batch(images.detach(), targets, label)
        self.writer.add_figure(tag, fig, global_step=global_step)
        plot.close()

    def log_dict(self, label, scalars, global_step):
        self.reset_if_needed(global_step)
        for key in scalars:
            tag = f"{label}/{key}" if label else key
            self.writer.add_scalar(tag, scalars[key], global_step=global_step)

    def log_config(self, config):
        config_dict = vars(config)
        hparams = { key:str(config_dict[key]) for key in config_dict }
        self.writer.add_hparams(hparams, {'dummy_metric' : 0.})

    def log_config_as_text(self, config, engine='str'):
        """

        :param config:
        :param engine: 'str' or 'jsonpickle'
        :return:
        """
        if engine == 'str':
            config_dict = vars(config)
            hparams = { key:str(config_dict[key]) for key in config_dict }
            df = pd.DataFrame.from_dict(hparams, orient='index')
        else:
            config_str = jsonpickle.encode(vars(config), indent=2)
            df = pd.read_json(config_str, orient='index')
        text = df.to_markdown()
        self.writer.add_text('config', text)

    def add_tensorboard_stuff(self, sparsity_mode, images, losses, probs,
                              sparsity, global_step, add_images=True):
        self.reset_if_needed(global_step)
        if add_images:
            #self.writer.add_images(f"{sparsity_mode}/Unfiltered Images", images, dataformats="NCHW",
            #        global_step=global_step)
            self.add_image_grid(images, f"{sparsity_mode}/Unfiltered Images",
                                filtered=False, num_per_row=3, global_step=global_step)
            #add_figure(images, f"{sparsity_mode}/Unfiltered Images", global_step, sparsity_mode)
            filtered_images = mh.mnist_post_process_image_batch(images)
            #add_figure(filtered_images, f"{sparsity_mode}/Filtered Images", global_step, sparsity_mode)
            self.add_image_grid(filtered_images, f"{sparsity_mode}/Filtered Images",
                                filtered=True, num_per_row=3, global_step=global_step)
            #self.writer.add_images(f"{sparsity_mode}/Filtered Images", filtered_images, dataformats="NCHW",
            #        global_step=global_step)
        self.log_dict(f"{sparsity_mode}/0-losses", losses, global_step)
        self.log_dict(f"{sparsity_mode}", probs, global_step)
        self.log_dict(f"{sparsity_mode}", sparsity, global_step)

    # Adds a list as text
    def add_list(self, lst, tag, num_per_row, global_step):
        self.reset_if_needed(global_step)
        n = len(lst)
        #text = ''
        chunked_lst = []
        lst = [ str(item) for item in lst ]
        for i in range(0, n, num_per_row):
            end = i+num_per_row
            end = end if end < n else n
            #chunk = ", ".join(lst[i:end])
            #text += chunk + "  \n"
            chunked_lst.append(lst[i:end])
        #print(text)
        #self.writer.add_text(tag, text, global_step)
        df = pd.DataFrame(chunked_lst)
        table = df.to_markdown()
        #print(table)
        self.writer.add_text(tag, table, global_step)

