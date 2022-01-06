import os
import torch
import argparse
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import torchsnooper
from tensorboard_logger import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict

from model import Encoder
from wgan_gp import GoodGenerator
from pix2pixnet import ResnetGenerator, NLayerDiscriminator
from loss import DiscriminatorLoss2, GeneratorLoss2
from data import Tradesy
from utils import *


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset',  help='The name of the dataset', default='tradesy')
    parser.add_argument('--sketch_dir', dest='sketch_dir', help='The direction of the sketch image',
                        default='/data/kmaeii/dataset/tradesy/tradesy_e2/train/sketch')
    parser.add_argument('--contour_dir', dest='contour_dir', help='The direction of the contour image',
                        default='/data/kmaeii/dataset/tradesy/tradesy_e2/train/contour')
    parser.add_argument('--mask_dir', dest='mask_dir', help='The direction of the mask image',
                        default='/data/kmaeii/dataset/tradesy/tradesy_e2/train/mask_erosion')
    parser.add_argument('--vgg_path', dest='vgg_path', help='the pre_model of the vgg16',default='./model/vgg16_bn-6c64b313.pth')
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=1)
    parser.add_argument('--num_works', dest='num_works', type=int, default=4)
    parser.add_argument('--output_dir',dest='output_dir', help='the direction of out put',default='./output')
    parser.add_argument('--log_dir',dest='log_dir', help='the direction of tensorboard_logger infomation',default='./runs')
    parser.add_argument('--restore', dest='restore', help='if restore the pre-train model', type=bool, default=False)

    parser.add_argument('--epoch', dest='epoch', help='num of epoch', type=int, default=1000)
    parser.add_argument('--batch_size', dest='batch_size', help='num of batch size', type=int, default=30)
    parser.add_argument('--zdim', dest='zdim', help='the dim of z', type=int, default=128)
    parser.add_argument('--num_class', dest='num_class', help='the class number of the data set', type= int, default=5)
    parser.add_argument('--num_block', dest='num_block', help='the resnet block number of the G net', type= int, default=9)
    parser.add_argument('--num_image', dest='num_image', help='the image number of the epoch save', type= int, default=64)
    parser.add_argument('--input_size', dest='input_size', help='the input image size of the network', default=128)

    parser.add_argument('--lr', dest='lr', help='The learning rate of the optimizer', type=float, default=2.0e-4)
    parser.add_argument('--beta1', dest='beta1', help='The beta1 param of the Adam optimizer', type=float, default=0.5)
    parser.add_argument('--lambda1', dest='lambda1', help='the weight of KL loss', type=float, default=1)
    parser.add_argument('--lambda2', dest='lambda2', help='the weight of G loss', type=float, default=1)


    parser.add_argument('--pre_e', dest='pre_e', help='the pre_model of the Encoder',
                        default='./model/E_epoch_89.pth')
    parser.add_argument('--pre_g1', dest='pre_g1', help='the pre_model of the Generator1',
                        default='./model/G1_epoch_89.pth')

    args = parser.parse_args()
    return args



def train(dataloader, model, loss_fun, optimizer, args):
    for epoch in tqdm(range(args.epoch), desc='Epoch'):
        model.g2.train()
        model.d2.train()
        for i, data in enumerate(dataloader):
            x_contour = data[1].cuda()
            label = data[3]
            assert not torch.any(torch.isnan(x_contour)), 'contour data has nan'
            assert not torch.any(torch.isnan(label)), 'label data has nan'
            # to one hot
            label_oh = label_to_one_hot(label.detach().numpy(), args.num_class)
            lable_r = torch.from_numpy(label_oh).float().cuda()

            #  ------------------------------Algorithm--------------------------------------------------------

            # 1. update  D2------------------------------------------------------------------------------------------
            optimizer.d2.zero_grad()

            mu, logvar = model.e(x_contour, lable_r)
            z = samper_z(mu, logvar)
            x_f = model.g1(z, lable_r)
            x_f2 = model.g2(x_f)

            # D2_loss #####
            D_out_r = model.d2(x_contour)
            D_r = D_out_r[-1]
            D_out_f = model.d2(x_f2.detach())
            D_f = D_out_f[-1]
            D2_loss = loss_fun.d2(D_r, D_f)
            D2_loss.backward()
            print('D2_loss : {}\n'.format(D2_loss.item()))
            args.logger.log_value('D2_loss', D2_loss.item())
            optimizer.d2.step()



            # 3. update  G2----------------------------------------------------------------------------------------------
            optimizer.g2.zero_grad()

            # G2_loss #####
            D_out_f = model.d2(x_f2)
            D_out_r = model.d2(x_contour)

            G2_loss = loss_fun.g2(D_out_r, D_out_f, x_contour, x_f2)
            print('G2_loss : {}\n'.format(G2_loss.item()))
            args.logger.log_value('G2_loss', G2_loss.item())

            G2_loss.backward()
            optimizer.g2.step()


        if epoch % 2 == 0: # test g in every 2 epoch
            epoch_save2(model, args, epoch)



def main():
    args = parser_args()
    args.logger = Logger(logdir=args.log_dir, flush_secs=1)
    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # make output_normal_tanh directiry
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load dataset
    if args.dataset == 'tradesy':
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
        dataset = Tradesy(args.sketch_dir, args.contour_dir,args.mask_dir,
                          transforms_sketch, transforms_contour, transforms_mask)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)
    else:
        raise Exception('Do not suppose this dataset')


    # build model
    model = {}
    model['e'] = Encoder(args.vgg_path, args.zdim, args.num_class).cuda()
    model['g1'] = GoodGenerator(dim=64, output_dim=args.input_size*args.input_size*1).cuda()

    model['g2'] = ResnetGenerator(input_nc=1, output_nc=1, ngf=64, n_blocks=args.num_block).cuda()
    model['d2'] = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, getIntermFeat=True).cuda()

    model = EasyDict(model)
    sd_e = torch.load(args.pre_e)
    sd_g1 = torch.load(args.pre_g1)
    model.e.load_state_dict(sd_e)
    model.g1.load_state_dict(sd_g1)
    if args.restore:
        sd_g2 = torch.load('')
        sd_d2 = torch.load('')
        model.g2.load_state_dict(sd_g2)
        model.d2.load_state_dict(sd_d2)
    else:
        pix_init_weights(model.d2)
        pix_init_weights(model.g2)

    # build loss function
    loss_fun = {}
    loss_fun['d2'] = DiscriminatorLoss2()
    loss_fun['g2'] = GeneratorLoss2(args)
    loss_fun = EasyDict(loss_fun)

    # build optimizer
    optimizer = {}
    optimizer['g2'] = optim.Adam(model.g2.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer['d2'] = optim.Adam(model.d2.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer = EasyDict(optimizer)

    train(dataloader, model, loss_fun, optimizer, args)


if __name__ == '__main__':
    main()