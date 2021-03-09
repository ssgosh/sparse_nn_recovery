if __name__ == '__main__':
    import sys
    sys.path.append('.')

import torch
from utils.ckpt_saver import CkptSaver
from models.mnist_mlp import MLPNet3Layer


def test_ckpt_save():
    ckpt = CkptSaver("ckpt_test/ckpt")
    images_list = [ torch.randn(20, 28, 28) for i in range(10) ]
    targets_list = [ torch.randint(high=10, low=0, size=(20,)) for i in range(10) ]
    probs_list = [ torch.randn(size=(20,10)) for i in range(10) ]
    for dataset_epoch, (images, targets, probs) in enumerate(zip(images_list, targets_list, probs_list)):
        ckpt.save_images('train', images, targets, probs, dataset_epoch)

    for dataset_epoch in range(10):
        images, targets, probs = ckpt.load_images('train', dataset_epoch)
        assert torch.all(images_list[dataset_epoch] == images).item()
        assert torch.all(targets_list[dataset_epoch] == targets).item()
        assert torch.all(probs_list[dataset_epoch] == probs).item()

    # Test model saving
    #model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5))
    model_class = MLPNet3Layer
    model_classname = 'MLPNet3Layer'
    model = model_class()
    for epoch in range(5):
        ckpt.save_model(model, epoch)
        model1 = ckpt.load_model_from_epoch(model_class, epoch)
        params1 = { n:p for n, p in model1.named_parameters() }
        for n, p in model.named_parameters():
            assert torch.all(params1[n] == p).item()

    for epoch in range(5, 10):
        ckpt.save_model(model, epoch, model_classname)
        model1 = ckpt.load_model_from_epoch(model_class, epoch, model_classname)
        params1 = { n:p for n, p in model1.named_parameters() }
        for n, p in model.named_parameters():
            assert torch.all(params1[n] == p).item()


if __name__ == '__main__':
    test_ckpt_save()

