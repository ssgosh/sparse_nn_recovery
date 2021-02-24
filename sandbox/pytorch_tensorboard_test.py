import math

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

def tensorboard_setup():
    writer = SummaryWriter()
    return writer

def main():
    x = torch.rand(10, 28, 28, requires_grad=True)
    target = torch.diagflat(torch.ones(28))
    target1 = torch.flip(target, dims=[1, ])
    target = torch.stack(5 * [target, target1])
    print(target.shape)

    opt = torch.optim.SGD([x], lr=1.0, momentum=0.9)
    for i in range(200):
        opt.zero_grad()
        diff = x - target
        mse = torch.mean(diff * diff)
        mse.backward()
        opt.step()

        print(f"step: {i}, loss: {mse.item():.3f}")
        writer.add_images("Images being trained", torch.unsqueeze(x, dim=1), dataformats="NCHW", global_step=i)
        writer.add_scalar("loss/training", mse.item(), global_step=i)

        img_grid = torchvision.utils.make_grid(torch.unsqueeze(x, dim=1), 4)
        writer.add_image("Image Grid", img_grid, global_step=i)

def foobar():
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    model = torchvision.models.resnet18(False)
    # Have ResNet model take in grayscale rather than RGB
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    images, labels = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images)
    writer.add_text('lstm', 'This is an lstm', 0)
    writer.add_text('rnn', 'This is an rnn', 10)
    layout = {'Taiwan':{'twse':['Multiline',['twse/0050', 'twse/2330']]},
                 'USA':{ 'dow':['Margin',   ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                      'nasdaq':['Margin',   ['nasdaq/aaa', 'nasdaq/bbb', 'nasdaq/ccc']]}}

    writer.add_custom_scalars(layout)
    writer.close()

def test_scalars():
    writer = SummaryWriter()

    for x in range(100):
        writer.add_scalars("some/tag", {'x' : x, 'x*sin(x)' : x*math.sin(x)}, x)

    for x in range(100):
        writer.add_scalars("another/tag", {'x' : x, 'x*cos(x)' : x*math.cos(x)}, x)


test_scalars()

#foobar()
#writer = tensorboard_setup()
#main()

