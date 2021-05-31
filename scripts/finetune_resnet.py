import sys
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms

def train(model, loader, opt, device):
    model.train()
    for batch_idx, (images, targets) in enumerate(loader):
        opt.zero_grad()
        images = images.to(device)
        targets = targets.to(device)
        out = model(images)
        loss = F.nll_loss(F.log_softmax(out, dim=1), targets)
        loss.backward()
        opt.step()
        sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(images), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        sys.stdout.write('\r')

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Dataset related code
# Imagenet mean and std
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_transforms = transforms.Compose([transforms.ToTensor(), normalize])
train_ds = datasets.CIFAR10('./data', download=True, train=True, transform=train_transforms)
test_ds = datasets.CIFAR10('./data', download=True, train=False, transform=test_transforms)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)

#for images, targets in train_loader:
#    print(images.shape, targets.shape)

device = torch.device('cuda:0')
model = models.resnet18(pretrained=True).to(device)
# Freeze model
for param in model.parameters():
    param.requires_grad = False

# For cifar, only 10 classes. So replace this layer
model.fc = torch.nn.Linear(512, 10).to(device)
opt = torch.optim.Adam(model.fc.parameters(), lr=1e-4, weight_decay=5e-4)
for epoch in range(10):
    train(model, train_loader, opt, device)
    test(model, test_loader, device)


