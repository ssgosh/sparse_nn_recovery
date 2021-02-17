import utils.image_processor as imp
import utils.mnist_helper as mh

import math


def _accumulate_val_in_dict(d, key, val):
    d[key] = (d[key] + val) if key in d else val


class MetricsHelper:
    def __init__(self, image_zero, image_one):
        self.image_zero = image_zero
        self.image_one = image_one

    # Computes average sparsity for each class in batch of images
    # Also for each layer for each class in batch
    def compute_sparsities(self, images, model, targets, sparsity):
        n = targets.shape[0]
        batch_sparsity = imp.get_sparsity_batch(images, zero=self.image_zero)
        count = {}
        for i in range(n):
            _accumulate_val_in_dict(count, targets[i], 1)
            key = f"1-class_{targets[i]}/sparsity/image"
            val = batch_sparsity[i].item()
            _accumulate_val_in_dict(sparsity, key, val)

        for tgt in count:
            key = f"1-class_{tgt}/sparsity/image"
            sparsity[key] /= count[tgt]

        for j, layer in enumerate(model.get_layers()):
            batch_sparsity = imp.get_sparsity_batch(layer, zero=0.)
            count = {}
            for i in range(n):
                _accumulate_val_in_dict(count, targets[i], 1)
                key = f"1-class_{targets[i]}/sparsity/layer_{j + 1}"
                val = batch_sparsity[i].item()
                _accumulate_val_in_dict(sparsity, key, val)

            for tgt in count:
                key = f"1-class_{tgt}/sparsity/layer_{j + 1}"
                sparsity[key] /= count[tgt]

    # Compute and store average probability of each class in the batch
    def compute_probs(self, output, probs, targets):
        count = {}
        for idx, tgt in enumerate(targets):
            prob = pow(math.e, output[idx][tgt.item()].item())
            key = f"1-class_{tgt}/prob"
            _accumulate_val_in_dict(probs, key, prob)

        for tgt in count:
            key = f"1-class_{tgt}/prob"
            probs[key] /= count[tgt]
