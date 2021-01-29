from __future__ import print_function
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plot
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np

from mnist_model import ExampleCNNNet
from mnist_mlp import MLPNet, MLPNet3Layer

def undo_transform(image):
    mean = 0.1307
    std = 0.3081
    return mean + image * std


def compute_generator_loss(config, model_all_l1, adv_output, adv_targetG): 
    lambd = config['lambd']
    lambd_layers = config['lambd_layers']
    include_likelihood = config['include_likelihood']
    include_layer = config['include_layer']

    if include_likelihood:
        nll_loss = F.nll_loss(adv_output, adv_targetG)
    else:
        nll_loss = 0.

    # include l1 penalty only if it's given as true for that layer
    l1_loss = 0.
    if include_layer[0]:
        l1_loss = lambd * (torch.norm(images + 0.5, 1)
                / torch.numel(images))

    l1_layers = 0.
    for include, lamb, l1 in zip(include_layer[1:], lambd_layers,
            model_all_l1):
        if include:
            l1_layers += lamb * l1

    loss = nll_loss + l1_loss + l1_layers
    return loss

# Performs the training steps for the discriminator
def training_step_discriminator(optD, data, target, adv_data, adv_targetF,
        model, config):
    #lambd = config['lambd']
    #lambd_layers = config['lambd_layers'] #[0.1, 0.1, 0.1]

    # Steps for training model on real data batch
    # Zero out gradients accumulated in the model's params
    optD.zero_grad()
    
    # Real data output and losses
    output = model(data)
    lossDR = F.nll_loss(output, target)

    # Fake data output and losses for discriminator
    adv_output = model(adv_data)
    lossDF = F.nll_loss(adv_output, adv_targetF)
    
    lossGF = compute_generator_loss(config, model_all_l1, adv_output, adv_targetR)

    # Supervised loss is now for classifying real data correctly, as well
    # as adversarial data correctly
    supervised_loss = lossDR + lossDF
    loss.backward()

    # Step optD, which changes only the model's params
    optD.step()


# Performs the training steps for "generator", which is really the
# discriminator trained backwards
def training_step_generator(optG, model, adv_data, adv_targetR):
    # Zero-out gradients from previous loss computations
    optG.zero_grad()
    
    optG.step()

# Adversarially train a single epoch
#
# adversarial_train_loader points to 1k 28x28 tensors
# These are initialized to randomly generated mnist_transform(N(0, 0.1))
# continuously modified during training
def adversarial_train(args, model, device, train_loader,
        adversarial_train_loader, optD, optG, epoch):
    model.train()
    for batch_idx, (data, target), (adv_data, adv_targetR, adv_targetF) in \
            enumerate(zip(train_loader, adversarial_train_loader)):
        # Some stupid pytorch things
        data, target = data.to(device), target.to(device)
        adv_data, adv_targetR, adv_targetF = adv_data.to(device), \
                adv_targetR.to(device), adv_targetF.to(device)

        training_step_discriminator(optD, model, data, target, adv_data, adv_targetF)
        training_step_generator(optG, model, adv_data, adv_targetR):

        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.write('\r')
            if args.dry_run:
                break
    pass

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.write('\r')
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Modified PyTorch MNIST Example')

    # Training mode
    #
    # 'adversarial' does adversarial training, with a single loop for
    # adversarial examples
    #
    # 'normal' just does supervised training 
    parser.add_argument('--train-mode', type=str, default='normal',
            metavar='MODE',
            help='Training mode: adversarial or normal')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # From this tutorial:
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset
    # , transforms are applied on each batch dynamically. Hence data gets
    # augmented due to random transforms.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9,
            1.1), shear=None),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    #dataset1 = datasets.MNIST('./data', train=True, download=True)
    #pilimage, label = dataset1[0]
    #print(label)
    #pilimage.show()
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=train_transform)

    # Print out the l0 norm of the images
    #dataset1 = datasets.MNIST('./data', train=True, download=True,
    #                   transform=transforms.Compose([transforms.ToTensor()]))
    #lst = []
    #for i in range(len(dataset1)):
    #    image, label = dataset1[i]
    #    l0_norm = torch.sum(image != 0.).item() / 784.
    #    lst.append(l0_norm)
        #print("%.3f" % l0_norm)
    #norms = np.array(lst)
    #print("mean = %.3f, median = %.3f, std = %.3f" % ( norms.mean(),
    #    np.median(norms),
    #    norms.std()))
    #sys.exit(1)
    # Plot some images
    #
    #for i in range(10):
    #    # Show one image
    #    image, label = dataset1[0]
    #
    #    imshow(image[0], cmap='gray')
    #    plot.show()
    # imshow(undo_transform(image)[0], cmap='gray')
    # plot.show()
    
    #np_img = undo_transform(image)[0].numpy()
    #img = Image.fromarray(np.uint8(np_img * 255), 'L')
    #img.show()
    # Show one image
    #sys.exit(1)

    dataset2 = datasets.MNIST('./data', train=False,
                       transform=test_transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #model = ExampleCNNNet().to(device)
    #model = MLPNet().to(device)
    model = MLPNet3Layer().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_mlp_3layer.pt")


if __name__ == '__main__':
    main()

