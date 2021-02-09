import torchvision

from torch.utils.tensorboard import SummaryWriter


def compute_mnist_transform_low_high():
    return -0.4242129623889923, 2.821486711502075


# high-pass and low-pass filter for images
def post_process_images(images, mode='mean_median', low=None, high=None):
    n = images.shape[0]
    channel = 0
    for idx in range(n):
        image = images[idx][channel]
        if mode == 'mean_median':
            mean = image.mean()
            median = image.median()
            low = (median + mean) / 2
        elif mode == 'low_high':
            assert low is not None or high is not None
        else:
            raise ValueError("Invalid value provided for mode %s" % mode)

        if low:
            image[image <= low] = low
        if high:
            image[image >= high] = high

class TensorBoardHelper:

    def __init__(self):
        self.writer = SummaryWriter()

    def mnist_post_process_image_batch(self, images):
        transformed_low, transformed_high = compute_mnist_transform_low_high()
        copied_images = images.clone().detach()
        post_process_images(copied_images, mode='low_high',
                low=transformed_low,
                high=transformed_high)
        return copied_images

    def add_image_grid(self, images, tag, global_step):
        images = images.detach()
        # Scale image
        img_grid = torchvision.utils.make_grid(images, 3, normalize=True,
                scale_each=True)
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
        for key in scalars:
            self.writer.add_scalar(f"{label}/{key}", scalars[key], global_step=global_step)

    def add_tensorboard_stuff(self, label, model, images, losses, probs, global_step):
        #self.writer.add_images(f"{label}/Unfiltered Images", images, dataformats="NCHW",
        #        global_step=global_step)
        self.add_image_grid(images, f"{label}/Unfiltered Images", global_step)
        #add_figure(images, f"{label}/Unfiltered Images", global_step, label)
        filtered_images = self.mnist_post_process_image_batch(images)
        #add_figure(filtered_images, f"{label}/Filtered Images", global_step, label)
        self.add_image_grid(filtered_images, f"{label}/Filtered Images", global_step)
        #self.writer.add_images(f"{label}/Filtered Images", filtered_images, dataformats="NCHW",
        #        global_step=global_step)
        self.log_dict(label, losses, global_step)
        self.log_dict(label, probs, global_step)
