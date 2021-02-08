import torch

from torch.utils.tensorboard import SummaryWriter

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

writer = tensorboard_setup()
main()

