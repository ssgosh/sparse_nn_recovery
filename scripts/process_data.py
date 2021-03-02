import torch

fname = "images_0014.pt"
d = torch.load(fname)
images = d['images']
targets = d['targets']
n = targets.shape[0]
idx = torch.randperm(n)

valid_images = images[idx[0:1000]].detach().to("cpu").clone()
valid_targets = targets[idx[0:1000]].detach().to("cpu").clone()

test_images = images[idx[1000:]].detach().to("cpu").clone()
test_targets = targets[idx[1000:]].detach().to("cpu").clone()

torch.save({'images' : valid_images, 'targets' : valid_targets}, "valid.pt")
torch.save({'images' : test_images, 'targets' : test_targets}, "test.pt")

