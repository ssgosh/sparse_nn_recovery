import jsonpickle
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from icontract import require
from torch.utils.tensorboard import SummaryWriter

from core.tblabels import TBLabels
from datasets.dataset_helper_factory import DatasetHelperFactory
from utils.image_processor import post_process_image_batch
from utils.torchutils import get_cross


class TensorBoardHelper:

    def __init__(self, name=None):
        self.name = name
        self.writer = SummaryWriter(name)

        # Reset SummaryWriter after these many global_steps
        self.reset_steps = 5000
        self.next_reset = self.reset_steps

        self.image_zero, self.image_one = DatasetHelperFactory.get().get_transformed_zero_one()
        self.batch_image_zero = DatasetHelperFactory.get().get_zero_correct_dims()
        self.batch_image_one = DatasetHelperFactory.get().get_one_correct_dims()
        self.num_real_classes = DatasetHelperFactory.get().get_num_classes()
        self.mean, self.std = DatasetHelperFactory.get().get_mean_std_correct_dims(include_batch=True)
        self.shape = DatasetHelperFactory.get().get_each_entry_shape()

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

    def transform_image_batch_to_valid_pixel_range(self, images):
        return images * self.std + self.mean

    def add_image_grid(self, images, tag, filtered, num_per_row, global_step):
        self.reset_if_needed(global_step)
        images = images.detach()
        # XXX: Remove scaling inside the range, and instead perform transformation into the range (0, 1)
        # rng = (self.image_zero, self.image_one) if filtered else None
        # img_grid = torchvision.utils.make_grid(images, num_per_row, normalize=True,
        #         range=rng, padding=2, pad_value=1.0,
        #         scale_each=True)

        images = self.transform_image_batch_to_valid_pixel_range(images)
        img_grid = torchvision.utils.make_grid(images, num_per_row, pad_value=1.0, padding=2)

        # Resize image grid
        #print(images.shape)
        #print(img_grid.shape)
        #size = img_grid.shape[1:]
        #print(size)
        #size = [3 * item for item in size]
        #print(size)
        img_grid = torch.unsqueeze(img_grid, 0)
        #print(img_grid.shape)
        # Scales the image to be a bit larger
        img_grid = F.interpolate(img_grid, scale_factor=10.0).squeeze(0)
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
                                filtered=False, num_per_row=10, global_step=global_step)
            #add_figure(images, f"{sparsity_mode}/Unfiltered Images", global_step, sparsity_mode)
            # filtered_images = post_process_image_batch(images, self.batch_image_zero, self.batch_image_one)
            # #add_figure(filtered_images, f"{sparsity_mode}/Filtered Images", global_step, sparsity_mode)
            # self.add_image_grid(filtered_images, f"{sparsity_mode}/Filtered Images",
            #                     filtered=True, num_per_row=3, global_step=global_step)
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

    @require(lambda data_type : data_type)
    def log_regular_batch_stats(self, data_type, suffix, model, images_tensor, targets_tensor, include_layer_map, sparsity_mode, dataset_epoch, precomputed=False):
        prefix = TBLabels.RECOVERY_EPOCH #"recovery_epoch"
        img_label = f"{prefix}/{data_type}_dataset_images_{suffix}" if suffix else f"{prefix}/{data_type}_dataset_images"
        tgt_label = f"{prefix}/{data_type}_dataset_targets_{suffix}" if suffix else f"{prefix}/{data_type}_dataset_targets"

        if precomputed:
            images, targets = images_tensor, targets_tensor
        else:
            images, targets = self.get_regular_batch(images_tensor, targets_tensor, self.num_real_classes, 10)
        targets_list = [foo.item() for foo in targets]
        self.add_image_grid(images, img_label, filtered=True, num_per_row=10,
                                global_step=dataset_epoch)
        self.add_list(targets_list, tgt_label, num_per_row=10,
                          global_step=dataset_epoch)
        # Run forward on this batch and get losses, probabilities and sparsity for logging
        # XXX: Skip this - we're going to do this on the full train data in
        #      AdversarialTrainer::generate_m_images_train_one_epoch_adversarial
        #
        # loss, losses, output, probs, sparsity = self.sparse_input_recoverer.forward(
        #     model, images, targets, include_layer_map[sparsity_mode], include_likelihood=True)
        # self.tbh.log_dict(f"{prefix}", probs, global_step=dataset_epoch)
        # self.tbh.log_dict(f"{prefix}", sparsity, global_step=dataset_epoch)
        self.flush()

    # Get a batch of 100 images with 10 images per class
    def get_regular_batch(self, images, targets, num_classes, num_per_class):
        entries = []
        tgt_entries = []
        for cls in range(num_classes):
            count = 0
            i = 0
            while count < num_per_class and i < targets.shape[0]:
                if targets[i].item() == cls:
                    entries.append(images[i])
                    tgt_entries.append(targets[i])
                    count += 1
                i += 1
            # Append cross X images if not enough entries for this class
            # All-zero images can be produced easily by our optimization algo,
            # But cross image is hard to be produced by accident
            for j in range(count, num_per_class):
                cross = get_cross(self.shape[2], self.shape[0], targets)

                entries.append(cross * self.batch_image_one.squeeze(dim=0) + self.batch_image_zero.squeeze(dim=0))
                tgt_entries.append(torch.tensor(cls, device=targets.device))

        return torch.stack(entries), torch.stack(tgt_entries)
