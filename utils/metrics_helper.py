import utils.image_processor as imp
import utils.mnist_helper as mh

import math
import json
import torch.nn.functional as F
import torch

from utils.dataset_helper import DatasetHelper


def _accumulate_val_in_dict(d, key, val):
    d[key] = (d[key] + val) if key in d else val


class MetricsHelper:
    @classmethod
    def get(cls):
        zero, one = DatasetHelper.get_dataset().get_transformed_zero_one()
        return cls(zero, one)

    def __init__(self, image_zero, image_one):
        self.image_zero = image_zero
        self.image_one = image_one

    def compute_avg_prob(self, output, target):
        real_probs = F.softmax(output, dim=1)
        a = torch.arange(target.shape[0])
        probs = real_probs[a, target]   # a[i], target[i] denotes desired class probability for ith entry
        avg_real_probs = torch.mean(probs)
        return avg_real_probs

    # Computes average sparsity for each class in batch of images
    # Also for each layer for each class in batch
    def compute_sparsities(self, images, model, targets, sparsity):
        n = targets.shape[0]
        batch_sparsity = imp.get_sparsity_batch(images, zero=self.image_zero)
        count = {}
        for i in range(n):
            _accumulate_val_in_dict(count, targets[i].item(), 1)
            key = f"class_{targets[i]}/sparsity/image"
            val = batch_sparsity[i].item()
            _accumulate_val_in_dict(sparsity, key, val)

        #print(json.dumps(sparsity, indent=2))
        for tgt in count:
            key = f"class_{tgt}/sparsity/image"
            sparsity[key] /= count[tgt]

        #print(json.dumps(sparsity, indent=2))
        #print(json.dumps(count, indent=2))
        for j, layer in enumerate(model.get_layers()):
            batch_sparsity = imp.get_sparsity_batch(layer, zero=0.)
            count = {}
            for i in range(n):
                _accumulate_val_in_dict(count, targets[i].item(), 1)
                key = f"class_{targets[i]}/sparsity/layer_{j + 1}"
                val = batch_sparsity[i].item()
                _accumulate_val_in_dict(sparsity, key, val)

            for tgt in count:
                key = f"class_{tgt}/sparsity/layer_{j + 1}"
                sparsity[key] /= count[tgt]
        #print(json.dumps(sparsity, indent=2))
        #print(json.dumps(count, indent=2))


    # Compute and store average probability of each class in the batch
    def compute_probs(self, output, probs, targets):
        count = {}
        for idx, tgt in enumerate(targets):
            _accumulate_val_in_dict(count, tgt.item(), 1)
            prob = pow(math.e, output[idx][tgt.item()].item())
            key = f"class_{tgt}/prob"
            _accumulate_val_in_dict(probs, key, prob)
            key = "avg_prob"
            _accumulate_val_in_dict(probs, key, prob)

        probs["avg_prob"] /= idx # Find the average prob over entire batch
        #print(json.dumps(probs, indent=2))
        #print(json.dumps(count, indent=2))
        for tgt in count:
            key = f"class_{tgt}/prob"
            probs[key] /= count[tgt]

        #print(json.dumps(probs, indent=2))