import utils.image_processor as imp
import utils.mnist_helper as mh


# Computes sparsity for each element in batch of images
# Also for each layer for each element of batch
def compute_sparsities(images, model, targets, sparsity):
    n = targets.shape[0]
    mnist_zero, _ = mh.compute_mnist_transform_low_high()
    batch_sparsity = imp.get_sparsity_batch(images, zero=mnist_zero)
    for i in range(n):
        sparsity[f"1-class_{targets[i]}/sparsity/image"] = batch_sparsity[i].item()

    for j, layer in enumerate(model.get_layers()):
        batch_sparsity = imp.get_sparsity_batch(layer, zero=0.)
        for i in range(n):
            sparsity[f"1-class_{targets[i]}/sparsity/layer_{j+1}"] = batch_sparsity[i].item()

