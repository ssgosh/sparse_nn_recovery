import torch

# high-pass and low-pass filter for images
import torchvision
import torch.nn.functional as F
from icontract import require

def save_grid_of_images(filename, images, targets, dh, sf):
    # dh.get_regular_batch()
    images, targets = dh.get_regular_batch(images, targets, dh.get_num_classes(), 10)
    images = dh.undo_transform_images(images)
    img_grid = torchvision.utils.make_grid(images, nrow=10, pad_value=1.0, padding=2)
    img_grid = torch.unsqueeze(img_grid, 0)
    img_grid = F.interpolate(img_grid, scale_factor=sf).squeeze(0)

    torchvision.utils.save_image(img_grid, filename)


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


def post_process_images_list(images_list, low, high):
    post_processed_images_list = []
    for images in images_list:
        #post_process_images(images)
        copied_images = images.clone().detach()
        #post_process_images(copied_images, mode='low_high', low=-0.5, high=2.0)
        post_process_images(copied_images, mode='low_high',
                low=low,
                high=high)
        post_processed_images_list.append(copied_images)

    return post_processed_images_list


# Computes sparsity of each image in the batch separately
# @require(lambda images: images.shape[1] == 1) # Only supports single channel as of now
def get_sparsity_batch(images, zero):
    n = len(images.shape)
    assert n > 1
    with torch.no_grad():
        return torch.sum(images > zero, dim=list(range(1, n)))


@require(lambda images: images.shape[1] == 1) # Only supports single channel as of now
def post_process_image_batch(images, transformed_low, transformed_high):
    # transformed_low, transformed_high = compute_mnist_transform_low_high()
    copied_images = images.detach().clone()
    clip_tensor_range(copied_images, batched_image_zero, batched_image_one, out)

    # post_process_images(copied_images, mode='low_high',
    #         low=transformed_low,
    #         high=transformed_high)
    return copied_images
