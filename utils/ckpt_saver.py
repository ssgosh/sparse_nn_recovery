import torch
import pathlib


class CkptSaver:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = pathlib.Path(ckpt_dir)

    def get_image_ckpt_path(self, mode, suffix, dataset_epoch):
        assert mode in ['train', 'test', 'valid', 'all']
        suffix = '' if not suffix else f'{suffix}_'
        return self.ckpt_dir / 'images' / mode / f'images_{suffix}{dataset_epoch:0>4d}.pt'

    def save_images(self, mode, suffix, images, targets, probs, sparsity, dataset_epoch):
        path = self.get_image_ckpt_path(mode, suffix, dataset_epoch)
        print("Writing images to : ", path)
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'images': images, 'targets': targets, 'probs': probs, 'sparsity' : sparsity}, path)

    def load_images(self, mode, dataset_epoch, device=None):
        path = self.get_image_ckpt_path(mode, dataset_epoch)
        print("Loading images from : ", path)
        model_dict = torch.load(path, map_location=device)
        # print(model_dict)
        return model_dict['images'], model_dict['targets'], model_dict['probs']

    # Checkpoint path for the model.
    # Expects model to be saved after each epoch
    # Suffix will typically be model class name
    def get_model_ckpt_path(self, epoch, suffix):
        suffix = f"{suffix}_" if suffix else ""
        return self.ckpt_dir / 'model' / f'model_{suffix}{epoch:0>4d}.pt'

    def get_ckpt_path(self, key, epoch, suffix):
        suffix = f"{suffix}_" if suffix else ""
        return self.ckpt_dir / f'{key}' / f'{key}_{suffix}{epoch:0>4d}.pt'

    def save_model(self, model, epoch, suffix=None):
        save_path = self.get_model_ckpt_path(epoch, suffix)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        print("Saving model to : ", save_path)
        assert not save_path.exists()
        torch.save(model.state_dict(), save_path)

    def load_model_from_path(self, model_class, path):
        pass

    def load_model_from_epoch(self, model_class, epoch, suffix=None):
        save_path = self.get_model_ckpt_path(epoch, suffix)
        print("Loading model from : ", save_path)
        state_dict = torch.load(save_path)
        model = model_class()
        model.load_state_dict(state_dict)
        return model

    def load_latest_model(self, model_class):
        pass

    def load_best_model(self, model_class):
        pass

    def save_key(self, obj, key, epoch, suffix=None):
        save_path = self.get_ckpt_path(key, epoch, suffix=suffix)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving {key} to : ", save_path)
        torch.save(obj.state_dict(), save_path)

    def save_everything(self, model, opt, sched, metrics, epoch, suffix=None):
        key = 'model_opt_sched'
        save_path = self.get_ckpt_path(key, epoch, suffix=suffix)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving {key} to : ", save_path)
        assert not save_path.exists()
        torch.save({
            'model' : model.state_dict(),
            'optimizer' : opt.state_dict(),
            'scheduler' : sched,
            'epoch' : epoch,
            'metrics' : metrics
        }, save_path)

    def load_evertything(self, model, optimizer, scheduler, resume_epoch, suffix=None):
        key = 'model_opt_sched'
        load_path = self.get_ckpt_path(key, resume_epoch, suffix=suffix)
        print(f'Loading model, optimizer, scheduler from : {load_path}')
        all_dict = torch.load(load_path)
        assert all_dict['epoch'] == resume_epoch, f"all_dict['epoch'] = {all_dict['epoch']}, resume_epoch = {resume_epoch}"
        print(f'Metrics : {all_dict["metrics"]}')
        model.load_state_dict(all_dict['model'])
        optimizer.load_state_dict(all_dict['optimizer'])
        scheduler = all_dict['scheduler']
        return model, optimizer, scheduler

