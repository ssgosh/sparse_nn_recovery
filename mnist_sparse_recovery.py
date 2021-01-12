from __future__ import print_function
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary
import matplotlib.pyplot as plot
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np

from mnist_model import Net

np.set_printoptions(precision = 3)

def undo_transform(image):
    mean = 0.1307
    std = 0.3081
    return mean + image * std

def recover_image(model, num_steps):
    image = torch.randn(1, 1, 28, 28)
    print("Initial image mean, std, min, max: ", image.mean().item(),
            image.std().item(),
            image.min().item(), image.max().item())
    #print("Initial image: ", torch.sum(image[0][0]))
    imshow(image[0][0], cmap='gray')
    plot.show()
    #imshow(undo_transform(image)[0][0], cmap='gray')
    #plot.show()
    image.requires_grad = True
    optimizer = optim.SGD([image], lr=0.1)

    # Target is the "0" digit
    target = torch.tensor([5])

    lambd = 1.
    for i in range(1, num_steps+1):
        output = model(image)
        loss = F.nll_loss(output, target) + lambd * torch.norm(image + 2, 1)
        loss.backward()
        print("Loss: ", loss.item(), "image mean, std, min, max: ",
                image.mean().item(), image.std().item(), image.min().item(),
                image.max().item())
        optimizer.step()

    #print("Final image: ", torch.sum(image[0][0]))
    image.requires_grad = False
    image[image <= -1.95] = -1.95
    image[image >=  1.95] =  1.95
    print("Final image mean, std, min, max: ", image.mean().item(),
            image.std().item(),
            image.min().item(), image.max().item())
    imshow(image[0][0], cmap='gray')
    plot.show()
    #imshow(undo_transform(image)[0][0], cmap='gray')
    #np_img = undo_transform(image)[0][0].numpy()
    #np_img[np_img < 0.] = 0.
    #print(image.mean())
    #print(image.std())
    #img = Image.fromarray(np.uint8(np_img * 255), 'L')
    #img.show()

model = Net()
model_state_dict = torch.load('mnist_cnn.pt')
model.load_state_dict(model_state_dict)
#print(model)
#summary(model, (1, 28, 28))

recover_image(model, 100)

