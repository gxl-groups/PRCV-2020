import argparse
import os

import torch
from data_tr import trans_To255, Tradesy, DataLoader
import torchvision.transforms as transforms
from torchvision import utils
from model import Generator
from tqdm import tqdm
def generate(args, g_ema, loader_test):

    with torch.no_grad():
        g_ema.eval()
        for i, data in enumerate(loader_test):
            if args.pics <= 0 : break

            _, img_contour_test, _, _, _, _ = data
            end = min(args.pics, img_contour_test.shape[0])
            sample_contour = img_contour_test.to(args.device)
            sample_contour = sample_contour[:end]
            utils.save_image(
                sample_contour,
                '{}/contour_{}.png'.format(args.sample_dir, i),
                nrow=8,
                normalize=True,
                range=(-1, 1),
            )

            sample, _ = g_ema(sample_contour)
            utils.save_image(
                sample,
                '{}/out_{}.png'.format(args.sample_dir, i),
                nrow=8,
                normalize=True,
                range=(-1, 1),
            )

            args.pics -= img_contour_test.shape[0]

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--sample_dir', type=str, default='./test_sample')
    parser.add_argument('--pics', type=int, default=500)
    parser.add_argument('--ckpt', type=str, default="./checkpoint_3.5/440000.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    args.device = 'cuda'
    torch.cuda.set_device(2)

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
        print('make the {} dir'.format(args.sample_dir))

    # model made
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(args.device)

    # load ckpt
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    g_ema.load_state_dict(checkpoint['g_ema'])

    # get dataset
    transforms_sketch = transforms.Compose([
        transforms.Resize(128),
        trans_To255(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transforms_contour = transforms.Compose([
        transforms.Resize(128),
        trans_To255(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transforms_mask = transforms.Compose([
        transforms.Resize(128),
        trans_To255(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_test = Tradesy('/data/kmaeii/dataset/tradesy/tradesy_e2/test/sketch',
                           '/data/kmaeii/dataset/tradesy/tradesy_e2/test/contour',
                           '/data/kmaeii/dataset/tradesy/tradesy_e2/test/mask_erosion',
                           transforms_sketch, transforms_contour, transforms_mask)
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=6)

    generate(args, g_ema, loader_test)
