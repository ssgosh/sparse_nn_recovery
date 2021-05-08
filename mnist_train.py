from __future__ import print_function
import argparse
import jsonpickle
import sys

import matplotlib
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plot
from matplotlib.pyplot import imshow
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pathlib

from core.adversarial_training import AdversarialTrainer
from core.sparse_input_dataset_recoverer import SparseInputDatasetRecoverer
from core.sparse_input_recoverer import SparseInputRecoverer
from datasets.dataset_helper import DatasetHelper
from datasets.dataset_helper_factory import DatasetHelperFactory
from utils.batched_tensor_view_data_loader import BatchedTensorViewDataLoader
import utils.mnist_helper as mh
from utils import runs_helper as rh
from utils.ckpt_saver import CkptSaver


# Penalized L1 Loss for training adversarial images
from utils.tensorboard_helper import TensorBoardHelper


def compute_generator_loss(config, adv_data, adv_output, adv_targetG, model_all_l1):
    lambd = config['lambd']
    lambd_layers = config['lambd_layers']
    include_likelihood = config['generator_include_likelihood']
    include_layer = config['generator_include_layer']

    if include_likelihood:
        # Cross-entropy of adversarial image on real classes (0, 1, 2...)
        nll_loss = F.nll_loss(adv_output, adv_targetG)
    else:
        nll_loss = 0.

    # include l1 penalty only if it's given as true for that layer
    l1_loss = 0.
    if include_layer[0]:
        l1_loss = lambd * (torch.norm(adv_data - mh.get_mnist_zero(), 1)
                / torch.numel(adv_data))

    l1_layers = 0.
    for include, lamb, l1 in zip(include_layer[1:], lambd_layers,
            model_all_l1):
        if include:
            l1_layers += lamb * l1

    loss = nll_loss + l1_loss + l1_layers
    return loss


# Performs the training steps for the discriminator and the generator for
# adversarial training
def training_step_adversarial(config, model, optD, optG, data, target, adv_data, adv_targetD,
        adv_targetG):
    # Now train the "generator"
    optG.zero_grad()

    # Fake data output and losses for discriminator. Cross-entropy of
    # adversarial images on fake-0, fake-1 etc classes
    adv_outputG = model(adv_data)

    # This should be computed here before adv_data gets changed 
    # Recomputation needed to separate out the two compute graphs
    adv_outputD = model(adv_data.detach())
    lossDF = F.nll_loss(adv_outputD, adv_targetD)

    # Since we have adv_output here, better compute this as well instead of
    # doing this twice
    lossG = compute_generator_loss(config, adv_data, adv_outputG, adv_targetG, model.all_l1)

    lossG.backward()
    optG.step()

    # Steps for training model on real data batch
    # Zero out gradients accumulated in the model's params
    optD.zero_grad()

    # Real data output and losses. Cross-entropy on real target
    output = model(data)
    lossDR = F.nll_loss(output, target)

    # Supervised loss is now for classifying real data correctly, as well
    # as adversarial data correctly
    supervised_loss = lossDR + lossDF
    supervised_loss.backward()

    # Step optD, which changes only the model's params
    optD.step()

    return supervised_loss, lossDR, lossDF, lossG


# Adversarially train a single epoch
#
# adversarial_train_loader points to 1k 28x28 tensors
# These are initialized to randomly generated mnist_transform(N(0, 0.1))
# continuously modified during training
def adversarial_train(args, config, model, device, train_loader,
        adversarial_train_loader, optD, optG, epoch):
    model.train()
    for batch_idx, ((data, target), (adv_data, adv_targetD, adv_targetG)) in \
            enumerate(zip(train_loader, adversarial_train_loader)):
        # Some stupid pytorch things
        data, target = data.to(device), target.to(device)
        adv_data, adv_targetD, adv_targetG = adv_data.to(device), \
                adv_targetD.to(device), adv_targetG.to(device)

        loss, lossDR, lossDF, lossG = training_step_adversarial(config, model, optD, optG, data, target, adv_data, adv_targetD,
                adv_targetG)

        if batch_idx % args.log_interval == 0:
            sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLossDR:'
                    '{:.6f}\tLossDF: {:.6f}\tLossG: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lossDR.item(),
                lossDF.item(), lossG.item()))
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
        loss = F.nll_loss(output, target) + model.get_weight_decay()
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    include_layer = {
        "no penalty"    : [ False, False, False, False],
        "input only"    : [ True, False, False, False],
        "all layers"    : [ True, True, True, True],
        "layer 1 only"  : [ False, True, False, False],
        "layer 2 only"  : [ False, False, True, False],
        "layer 3 only"  : [ False, False, False, True],
        "all but input" : [ False, True, True, True],
        }
    generator_modes = list(include_layer.keys())

    parser = argparse.ArgumentParser(description='Modified PyTorch MNIST Example', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    available_train_modes = ['normal', 'adversarial-continuous', 'adversarial-epoch', 'adversarial-batches', 'test']
    parser.add_argument('--train-mode', type=str, default='normal', metavar='MODE', choices=available_train_modes,
                        help='Training mode. One of: ' + ', '.join(available_train_modes))
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='MODE')
    parser.add_argument('--non-sparse-dataset', action='store_true', default=True, dest='non_sparse_dataset', help='Load dataset in non-sparse mode')
    parser.add_argument('--sparse-dataset', action='store_false', default=True, dest='non_sparse_dataset', help='Load dataset in sparse mode')
    parser.add_argument('--pretrain', action='store_true', default=True, dest='pretrain', help='Pretrain before adversarial training')
    parser.add_argument('--no-pretrain', action='store_false', default=True, dest='pretrain', help='Do not pretrain')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing ')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train')
    parser.add_argument('--num-pretrain-epochs', type=int, default=1, metavar='N', help='number of epochs to pre-train before starting adversarial training')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, dest='save_model', help='For Saving the current Model')
    parser.add_argument('--no-save-model', action='store_false', default=True, dest='save_model', help='Do not save the current Model')
    parser.add_argument('--run-dir', type=str, default=None, metavar='DIR', help='Directory under which model and outputs are saved')
    parser.add_argument('--run-suffix', type=str, default='', required=False, metavar='S', help='Will be appended to the run directory provided')
    parser.add_argument('--early-epoch', action='store_true', default=False, dest='early_epoch', help='Finish epoch early (for debugging)')
    parser.add_argument('--num-batches-early-epoch', type=int, default=10, metavar='N', help='Number of batches before epoch finishes')
    parser.add_argument('--dump-config', action='store_true', default=False, required=False, help='Print config json and exit')
    parser.add_argument('--resume-epoch', type=int, default=None, required=False, help='Resume from checkpoint for this saved epoch')
    parser.add_argument('--load-model', action='store_true', default=False, required=False, help='Load model from default location')


    # Arguments specific to adversarial training
    parser.add_argument('--generator-lr', type=float, default=0.05,
                        metavar='GLR',
                        help='learning rate for image generation')

    parser.add_argument('--generator-mode', type=str, default='input only',
            metavar='GM',
            help='Generator penalty mode. One of: "' +
            '", "'.join(generator_modes) + '"')
    #parser.add_argument('--generator-lambda', type=float, default=0.1, metavar='LAMBDA',
    #                    help='lambda value for input layer')
    #parser.add_argument('--generator-lambda-layers', nargs=3, type=float, default=[0.1,
    #                    0.1, 0.1], metavar='a b c',
    #                    help='lambda value for input layer')
    #parser.add_argument('--generator-include-likelihood',
    #                    dest='generator_include_likelihood',
    #                    action='store_true', default=True,
    #                    help='include likelihood loss')
    #include_likelihood = config_dict['generator_include_likelihood']
    #include_layer = config_dict['generator_include_layer']

    AdversarialTrainer.add_command_line_arguments(parser)
    SparseInputDatasetRecoverer.add_command_line_arguments(parser)
    SparseInputRecoverer.add_command_line_arguments(parser)

    args = parser.parse_args()

    config = args

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config.use_cuda = use_cuda
    config.device = device

    SparseInputRecoverer.setup_default_config(config)
    # dataset name is 'MNIST'
    #config.dataset_name = 'mnist'
    dataset_helper: DatasetHelper = DatasetHelperFactory.get(config.dataset, config.non_sparse_dataset)
    dataset_helper.setup_config(config)

    # Setup runs directory, tensorboard helper and sparse input recoverer
    rh.setup_run_dir(config, 'train_runs')
    tbh = TensorBoardHelper(config.run_dir)
    sparse_input_recoverer = SparseInputRecoverer(config, tbh, verbose=True)
    config.ckpt_dir = f"{args.run_dir}/ckpt/"
    config.ckpt_save_path = pathlib.Path(f"mnist_cnn.pt")
    ckpt_saver = CkptSaver(config.ckpt_dir)
    # Log config to tensorboard
    tbh.log_config_as_text(config)
    tbh.flush()

    # Set config_dict from args
    config_dict = vars(args)
    config_dict['lambd'] = 0.1
    config_dict['lambd_layers'] = [0.1, 0.1, 0.1]
    config_dict['generator_include_likelihood'] = True
    config_dict['generator_include_layer'] = include_layer[args.generator_mode]

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 3,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # test_transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # From this tutorial:
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset
    # , transforms are applied on each batch dynamically. Hence data gets
    # augmented due to random transforms.
    # train_transform = transforms.Compose([
    #     transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9,
    #         1.1), shear=None),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    #dataset1 = datasets.MNIST('./data', train=True, download=True)
    #pilimage, label = dataset1[0]
    #print(label)
    #pilimage.show()
    # dataset1 = datasets.MNIST('./data', train=True, download=True,
    #                    transform=train_transform)
    #
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
    # imshow(mh.undo_transform(image)[0], cmap='gray')
    # plot.show()

    #np_img = mh.undo_transform(image)[0].numpy()
    #img = Image.fromarray(np.uint8(np_img * 255), 'L')
    #img.show()
    # Show one image
    #sys.exit(1)

    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # From this tutorial:
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset
    # , transforms are applied on each batch dynamically. Hence data gets
    # augmented due to random transforms.
    # train_transform = transforms.Compose([
    #     transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9,
    #                                                                     1.1), shear=None),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    # dataset1 = datasets.MNIST('./data', train=True, download=True,
    #                           transform=train_transform)
    #
    # dataset2 = datasets.MNIST('./data', train=False,
    #                    transform=test_transform)
    dataset1 = dataset_helper.get_dataset(which='train', transform='train')
    dataset2 = dataset_helper.get_dataset(which='test', transform='test')
    print(f"Dataset name : {config.dataset}, train_len = {len(dataset1)}, test_len = {len(dataset2)}")
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    # full_train_data = datasets.MNIST('./data', train=True, download=True,
    #                           transform=test_transform)
    full_train_data = dataset_helper.get_dataset(which='train', transform='test')
    train_samples = DataLoader(Subset(full_train_data, indices=torch.randperm(len(full_train_data))[0:10000]), **test_kwargs)

    #imshow(mh.undo_transform(image)[0], cmap='gray')
    #plot.show()

    if args.train_mode == 'adversarial-continuous':
        # 1000 images of size 28x28, 1 channel
        # initialize images with a Gaussian ball close to mnist 0
        #images = torch.normal(mnist_zero + 0.1, 0.1, (1000, 1, 28, 28), requires_grad=True)
        images = torch.randn(1000, 1, 28, 28, requires_grad=True)
        real_class_targets = torch.randint(10, (1000, ))
        # class fake-0 is 10, fake-1 is 11 etc
        fake_class_targets = real_class_targets + 10
        adversarial_dataset = torch.utils.data.TensorDataset(images,
                real_class_targets, fake_class_targets)
        #adversarial_train_loader = InfiniteDataLoader(adversarial_dataset,
        #        **train_kwargs)
        adversarial_train_loader = BatchedTensorViewDataLoader(args.batch_size,
                images, real_class_targets, fake_class_targets)
        batch_a, batch_b, batch_c = next(iter(adversarial_train_loader))
        #print(len(batch_a))
        #print(batch_a[0], batch_b[0], batch_c[0])
        #sys.exit(0)

    assert not (args.load_model and (args.resume_epoch is not None))
    load = False
    #if config.dataset.lower() == 'cifar':
    #    load = True
    #    # config.discriminator_model_file =
    model = dataset_helper.get_model(config.adversarial_classification_mode, device, load=args.load_model, config=config)
    optimizer, scheduler = dataset_helper.get_optimizer_scheduler(config, model)
    start_epoch = 0
    if args.resume_epoch is not None:
        model, optimizer, scheduler = ckpt_saver.load_evertything(model, optimizer, scheduler, args.resume_epoch)
        start_epoch = args.resume_epoch + 1

    if args.train_mode == 'adversarial-continuous':
        optD = optimizer
        optG = optim.Adam([images], lr=args.generator_lr)

    # Setup sparse dataset recovery here, after model etc are all set up
    if args.train_mode in [ 'adversarial-epoch', 'adversarial-batches' ]:
        if args.train_mode == 'adversarial-epoch':
            dataset_len = config.num_adversarial_images_epoch_mode
        elif args.train_mode == 'adversarial-batches':
            dataset_len = config.num_adversarial_images_batch_mode
        dataset_recoverer = SparseInputDatasetRecoverer(
            sparse_input_recoverer,
            model,
            num_recovery_steps=config.recovery_num_steps,
            batch_size=config.recovery_batch_size,
            sparsity_mode=config.recovery_penalty_mode,
            num_real_classes=dataset_helper.get_num_classes(),
            dataset_len=dataset_len,
            each_entry_shape=dataset_helper.get_each_entry_shape(),
            device=device, ckpt_saver=ckpt_saver, config=config)
        #images, targets = dataset_recoverer.recover_image_dataset()
        #print("Recovered images, targets", images.shape, targets.shape, targets.detach().numpy())
        #sys.exit(0)
        # Now we can create an AdversarialTrainer!!!!!
        adversarial_trainer = AdversarialTrainer(train_loader, train_samples, dataset_recoverer, model, optimizer, config.batch_size,
                                                 device, config.log_interval, config.dry_run, config.early_epoch,
                                                 config.num_batches_early_epoch, test_loader, scheduler,
                                                 config.adversarial_classification_mode, config)

    # Log config to tensorboard
    # tbh.log_config_as_text(config)
    # tbh.flush()
    # Dump configuration after setting everything up. For quick debugging
    config_str = jsonpickle.encode(vars(config), indent=2)
    with open(f"{config.run_dir}/config.json" , 'w') as f:
        f.write(config_str)
    if config.dump_config:
        #json.dump(vars(config), sys.stdout, indent=2, sort_keys=True)
        print(config_str)
        sys.exit(0)

    if args.train_mode == 'test':
        print('Testing only, no training')
        test(model, device, test_loader)
    elif args.train_mode not in ['adversarial-batches', 'adversarial-epoch']:
        print('Testing before training:')
        test(model, device, test_loader)
        for epoch in range(start_epoch, args.epochs):
            # Perform pre-training for 1 epoch in adversarial mode
            if args.train_mode == 'normal' or epoch == 0:
                if args.train_mode == 'adversarial-continuous':
                    print('Performing pre-training for 1 epoch')
                train(args, model, device, train_loader, optimizer, epoch)
            elif args.train_mode == 'adversarial-continuous':
                adversarial_train(args, config_dict, model, device, train_loader,
                                  adversarial_train_loader, optD, optG, epoch)
            else:
                raise ValueError("invalid train_mode : " + args.train_mode)
            test(model, device, test_loader)
            ckpt_saver.save_model(model, epoch, config.model_classname)
            ckpt_saver.save_everything(model, optimizer, scheduler, {}, epoch)
            scheduler.step()
    else:
        adversarial_trainer.train_loop(start_epoch, args.epochs, args.train_mode, args.pretrain, args.num_pretrain_epochs, config)

    if args.save_model:
        save_path = config.ckpt_save_path
        save_path.parent.mkdir(exist_ok=True, parents=True)
        print("Saving model to : ", save_path)
        torch.save(model.state_dict(), save_path)

    tbh.close()


if __name__ == '__main__':
    main()
    #import cProfile
    #cProfile.run('main()')

