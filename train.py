from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse
import os

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow
from grad_estimation import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=3, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=8, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("--path", metavar="PATH", type=str, help="Path to get results and models")
parser.add_argument("--path_dataset", required=True, help = "Path to directory to save images")
parser.add_argument("--model_previous", default=None)
parser.add_argument("--optim_previous", default=None)

parser.add_argument("save_sample_every", default = 1000)
parser.add_argument("save_checkpoint_every", default = 5000)
parser.add_argument("save_likelihood_every", default = 5000)




def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

def get_ImageFolder_dataset(path):
    dataset = datasets.ImageFolder(path, transform=default_transform)

def sample_data(dataset, batch_size, image_size):

    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)

def train(dataset, args, model, optimizer, path = "" , test_image_temoin = None, test_image_critic = None):
    if not os.path.exists(path):
      os.makedirs(path)


    dataset_loader = iter(sample_data(dataset, args.batch, args.img_size))

    if test_image_temoin is not None and test_image_critic is not None :
      testing = True
      test_image_temoin = test_image_temoin.to(device)
      test_image_critic = test_image_critic.to(device)
    else :
      testing = False

    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset_loader)
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if i % args.save_sample_every == 0:
                path_sample = os.path.join(path, 'sample')
                if not os.path.exists(path_sample):
                  os.makedirs(path_sample)
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        os.path.join(path_sample,f"{str(i + 1).zfill(6)}.png"),
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % args.save_checkpoint_every == 0:
                path_checkpoint = os.path.join(path, "checkpoint")
                if not os.path.exists(path_checkpoint):
                  os.makedirs(path_checkpoint)
                torch.save(
                    model.state_dict(), os.path.join(path_checkpoint,f"model_{str(i + 1).zfill(6)}.pt")
                )
                torch.save(
                    optimizer.state_dict(), os.path.join(path_checkpoint,f"optim_{str(i + 1).zfill(6)}.pt")
                )
            if i%args.save_likelihood_every ==0:
                path_likelihood = os.path.join(path, "likelihood")
                if not os.path.exists(path_likelihood):
                  os.makedirs(path_likelihood)
                save_likelihood(path_likelihood, i, model, test_image_temoin, test_image_critic, nb_step = 0)

            # if i % 100 == 0:
            #     with torch.no_grad():
            #         utils.save_image(
            #             model_single.reverse(z_sample).cpu().data,
            #             f"sample/{str(i + 1).zfill(6)}.png",
            #             normalize=True,
            #             nrow=10,
            #             range=(-0.5, 0.5),
            #         )

            # if i % 10000 == 0:
            #     torch.save(
            #         model.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
            #     )
            #     torch.save(
            #         optimizer.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
            #     )


if __name__ == "__main__":


    

    args = parser.parse_args()


        
    default_transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    # mnist_dataset = 
    cifar_dataset_train = datasets.CIFAR10(args.path_dataset, transform = default_transform, download=True)
    cifar_dataset_test = datasets.CIFAR10(args.path_dataset, transform = default_transform, train=False, download = True)
    dataloader_cifar_test = torch.utils.data.DataLoader(cifar_dataset_test, batch_size = 100)
    test_image_temoin = next(iter(dataloader_cifar_test))[0]

    svhn_dataset_test = datasets.SVHN(args.path_dataset, transform = default_transform, download = True)
    dataloader_svhn = torch.utils.data.DataLoader(svhn_dataset_test, batch_size = 100)
    test_image_critic = next(iter(dataloader_svhn))[0]




    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )

    if args.model_previous is not None :
        model.load_state_dict(torch.load(args.model_previous))

    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer_previous is not None :
        optimizer.load_state_dict(torch.load(args.optimizer_previous))

    
    train(cifar_dataset_train, args, model, optimizer, path = args.path, test_image_temoin = test_image_temoin, test_image_critic = test_image_critic)
