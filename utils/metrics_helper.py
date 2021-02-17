import utils.image_processor as imp
import utils.mnist_helper as mh

import math


class MetricsHelper:
    def __init__(self, image_zero, image_one):
        self.image_zero = image_zero
        self.image_one = image_one

    # Computes sparsity for each element in batch of images
    # Also for each layer for each element of batch
    def compute_sparsities(self, images, model, targets, sparsity):
        n = targets.shape[0]
        batch_sparsity = imp.get_sparsity_batch(images, zero=self.image_zero)
        for i in range(n):
            sparsity[f"1-class_{targets[i]}/sparsity/image"] = batch_sparsity[i].item()

        for j, layer in enumerate(model.get_layers()):
            batch_sparsity = imp.get_sparsity_batch(layer, zero=0.)
            for i in range(n):
                sparsity[f"1-class_{targets[i]}/sparsity/layer_{j + 1}"] = batch_sparsity[i].item()

    def compute_probs(self, output, probs, targets):
        for idx, tgt in enumerate(targets):
            prob = pow(math.e, output[idx][tgt.item()].item())
            # print(prob)
            probs[f"1-class_{tgt}/prob"] = prob
