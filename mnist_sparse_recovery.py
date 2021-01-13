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
import math

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
    plot.colorbar()
    #plot.draw()
    #plot.pause(0.0001)
    #plot.show()
    #imshow(undo_transform(image)[0][0], cmap='gray')
    plot.show()
    image.requires_grad = True
    optimizer = optim.Adam([image], lr=0.05)
    #optimizer = optim.SGD([image], lr=0.1)

    # Target is the "0" digit
    target = torch.tensor([0])

    # lambda for input
    lambd = 0.1
    # lambda for each layer
    lambd_layers = [0.1, 0.1, 0.1]
    #lambd2 = 1.
    for i in range(1, num_steps+1):
        optimizer.zero_grad()
        output = model(image)
        nll_loss = F.nll_loss(output, target)
        l1_loss = lambd * (torch.norm(image + 2., 1)
                / torch.numel(image))
        l1_layers = sum([ (lamb * l1) for lamb, l1 in zip(lambd_layers,
            model.all_l1) ])
        #l2_loss = lambd2 * (torch.norm(image, + 2) ** 2
        #        / torch.numel(image))

        #loss = nll_loss + l1_loss
        #loss = nll_loss + l1_layers 
        loss = nll_loss + l1_loss + l1_layers
        #loss = l1_layers
        #loss = nll_loss
        loss.backward()
        print("Iter: ", i,", Loss: %.3f" % loss.item(),
                f"Prob of {target[0]} %.3f" %
                pow(math.e, output[0][target[0].item()].item()),
                "image mean, std, min, max: %.3f, %.3f, %.3f, %.3f" % (
                image.mean().item(), image.std().item(), image.min().item(),
                image.max().item()))
        optimizer.step()
        #image.requires_grad = False
        #plot.clf()
        #imshow(image[0][0], cmap='gray')
        #plot.colorbar()
        #plot.draw()
        #plot.pause(0.0001)
        #image.requires_grad = True

    #print("Final image: ", torch.sum(image[0][0]))
    image.requires_grad = False
    mean = image.mean()
    image[image <= mean] = mean
    #image[image >=  1] =  1.
    #image[image <= -2] = -2.
    #image[image >=  1] =  1.
    print("Final image mean, std, min, max: %.3f, %.3f, %.3f, %.3f" % (
        image.mean().item(), image.std().item(), image.min().item(),
        image.max().item()))
    imshow(image[0][0], cmap='gray')
    plot.colorbar()
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

recover_image(model, 10000)

