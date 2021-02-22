import torch
import pathlib

class CkptSaver:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = pathlib.Path(ckpt_dir)

    def get_ckpt_path(self, dataset_epoch):
        return self.ckpt_dir / 'images' / f'images_{dataset_epoch:0>4d}.pt'

    def save_images(self, images, targets, dataset_epoch):
        path = self.get_ckpt_path(dataset_epoch)
        print(path)
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'images' : images, 'targets' : targets}, path)

    def load_images(self, dataset_epoch):
        path = self.get_ckpt_path(dataset_epoch)
        print(path)
        model_dict = torch.load(path)
        #print(model_dict)
        return model_dict['images'], model_dict['targets']

