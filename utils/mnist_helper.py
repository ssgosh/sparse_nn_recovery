from . import image_processor as imp


# Pre-computed from below commented-out function
# python3.6/pytorch1.3.0 had some issues with the commented function
def compute_mnist_transform_low_high():
    assert False, "This method is now deprecated in favour of DatasetHelper::get_transformed_zero_one()"
    return -0.4242129623889923, 2.821486711502075


#def compute_mnist_transform_low_high():
#    mean = 0.1307
#    std = 0.3081
#    transform = transforms.Normalize(mean, std)
#    low = torch.zeros(1, 1, 1)
#    high = low + 1
#    print(torch.sum(low).item(), torch.sum(high).item())
#    transformed_low = transform(low).item()
#    transformed_high = transform(high).item()
#    print(transformed_low, transformed_high)
#    return transformed_low, transformed_high


def get_mnist_zero():
    mnist_zero, mnist_one = compute_mnist_transform_low_high()
    return mnist_zero


def undo_transform(image):
    mean = 0.1307
    std = 0.3081
    return mean + image * std


def mnist_post_process_image_batch(images):
    transformed_low, transformed_high = compute_mnist_transform_low_high()
    copied_images = images.clone().detach()
    imp.post_process_images(copied_images, mode='low_high',
            low=transformed_low,
            high=transformed_high)
    return copied_images

def mnist_post_process_images_list(images_list):
    transformed_low, transformed_high = compute_mnist_transform_low_high()
    imp.post_process_images_list(images_list, transformed_low,
            transformed_high)

