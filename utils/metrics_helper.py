#from __future__ import annotations
from typing import List

import utils.image_processor as imp

import math
import torch.nn.functional as F
import torch

from core.mlabels import MLabels
from datasets.dataset_helper import DatasetHelperFactory
from utils.tensorboard_helper import TensorBoardHelper


def _accumulate_val_in_dict(d, key, val):
    d[key] = (d[key] + val) if key in d else val


def _safe_divide(y, x):
    """
    Safely divide y by x. Performs division for non-zero denominator elements only.

    Typically used to take per-class averages from per-class sums and per-class counts

    :param y: sums of some metric such as loss, with one entry per class
    :param x: number of occurrences of each class
    :return: Nothing. division done in-place in y
    """
    y[x != 0] = y[x != 0] / x[x != 0]

class MetricsHelper:
    @classmethod
    def get(cls, mlabels : MLabels = None, adversarial_classification_mode='max-entropy') -> 'MetricsHelper':
        zero, one = DatasetHelperFactory.get().get_transformed_zero_one()
        # XXX: Change this to get_num_classes. This is not valid in case of soft adversarial labels
        num_real_fake_classes = DatasetHelperFactory.get().get_num_real_fake_classes()
        return cls(zero, one, mlabels, num_real_fake_classes, adversarial_classification_mode)

    @classmethod
    def reduce(cls, metrics_list: List['MetricsHelper']):
        overall_metrics = cls.get()
        if not metrics_list: return overall_metrics
        overall_metrics.mlabels = metrics_list[0].mlabels
        n = len(metrics_list)
        overall_metrics.numel = sum([m.numel for m in metrics_list])
        overall_metrics.correct = sum([m.correct for m in metrics_list])
        for key in metrics_list[0].agg:
            overall_metrics.agg[key] = sum([m.agg[key] for m in metrics_list]) / n
        for key in metrics_list[0].per_class:
            overall_metrics.per_class[key] = sum([m.per_class[key] for m in metrics_list]) / n
        overall_metrics.initialized = True
        overall_metrics.finalized = True
        return overall_metrics

    def __init__(self, image_zero, image_one, mlabels : MLabels = None, num_real_fake_classes=None,
                 adversarial_classification_mode='max-entropy'):
        self.image_zero = image_zero
        self.image_one = image_one
        self.agg = {}
        self.per_class = {}
        self.correct = 0
        self.mlabels = mlabels
        self.adversarial_classification_mode = adversarial_classification_mode

        # Keep track of batch and number of elements
        # Updated on each call to accumulate_batch_stats
        self.num_batches = 0
        self.numel = 0
        self.per_class_numel = torch.zeros(size=(num_real_fake_classes,))

        self.initialized = False    # Someone has to add some metrics using accumulate_batch_stats() for this to happen.
        self.finalized = False

    def compute_reduce_prob(self, output, target, reduce='avg', adv_data=False):
        assert reduce in ['sum', 'avg']
        real_probs = F.softmax(output, dim=1)
        a = torch.arange(target.shape[0])
        if adv_data and self.adversarial_classification_mode == 'max-entropy':
            probs = torch.max(real_probs, dim=1)[0]
        else:
            probs = real_probs[a, target] # a[i], target[i] denotes desired class probability for ith entry
        if reduce == 'avg':
            reduction = torch.mean(probs)
        elif reduce == 'sum':
            reduction = torch.sum(probs)
        return reduction

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


    # Compute and store average probability of each class in the batch
    def compute_agg_per_class_probs_separately(self, output, targets, agg, per_class, mlabels):
        count = {}
        for idx, tgt in enumerate(targets):
            _accumulate_val_in_dict(count, tgt.item(), 1)
            prob = pow(math.e, output[idx][tgt.item()].item())
            key = f"class_{tgt}/prob"
            _accumulate_val_in_dict(per_class, key, prob)
            key = mlabels.avg_prob
            _accumulate_val_in_dict(agg, key, prob)

        agg[mlabels.avg_prob] /= idx  # Find the average prob over entire batch
        # print(json.dumps(probs, indent=2))
        # print(json.dumps(count, indent=2))
        for tgt in count:
            key = f"class_{tgt}/prob"
            per_class[key] /= count[tgt]

        # print(json.dumps(probs, indent=2))

    def accumulate_batch_stats(self, output, target, adv_data=False, adv_soft_labels=None):
        assert not self.finalized
        self.initialized = True
        self.numel += target.shape[0]
        if adv_data and self.adversarial_classification_mode == 'max-entropy':
            assert adv_soft_labels is not None
            loss = F.kl_div(output, adv_soft_labels, reduction='sum').item()
            probs = F.softmax(output, dim=1)
            max_probs = probs.max(dim=1)[0]
            # pred is True if adversarial data. If max prob amongst all classes is less than 0.5,
            # we consider the model sufficiently confused on adversarial data
            pred = max_probs < 0.5
            correct = pred.sum().item()
        else:
            loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
        self.correct += correct
        sum_probs = self.compute_reduce_prob(output, target, reduce='sum', adv_data=adv_data).item()
        _accumulate_val_in_dict(self.agg, self.mlabels.avg_loss, loss)
        _accumulate_val_in_dict(self.agg, self.mlabels.avg_accuracy, correct)
        _accumulate_val_in_dict(self.agg, self.mlabels.avg_prob, sum_probs)

    def log(self, name, tbh : TensorBoardHelper, tb_agg_label, tb_per_class_label, global_step):
        if not self.initialized : return
        assert self.finalized
        print(f'Test set {name}:',
              'Average loss: {:.4f}, '.format(self.agg[self.mlabels.avg_loss]),
              'Average Prob: {:.4f}, '.format(self.agg[self.mlabels.avg_prob]),
              'Accuracy: {}/{} ({:.2f}%)'.format(
                  self.correct, self.numel,
                  100. * self.correct / self.numel))
        tbh.log_dict(tb_agg_label, self.agg, global_step)
        tbh.log_dict(tb_per_class_label, self.per_class, global_step)

    # Divide by counts
    def finalize_stats(self):
        for key in self.agg:
            self.agg[key] /= self.numel
        self.finalized = True

