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

from model import Encoder, Classifier
from wgan_gp import GoodGenerator, GoodDiscriminator
from loss import DiscriminatorLoss1, GeneratorLoss1, ClassifierLoss
from data import Tradesy
from utils import *
import torchvision.models as models


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
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0)
    parser.add_argument('--num_works', dest='num_works', type=int, default=4)
    parser.add_argument('--output_dir',dest='output_dir', help='the direction of out put',default='./output_s1')
    parser.add_argument('--log_dir',dest='log_dir', help='the direction of tensorboard_logger infomation',default='./runs_s1')
    parser.add_argument('--restore', dest='restore', help='if restore the pre-train model', type=bool, default=False)

    parser.add_argument('--epoch', dest='epoch', help='num of epoch', type=int, default=1000)
    parser.add_argument('--batch_size', dest='batch_size', help='num of batch size', type=int, default=22)
    parser.add_argument('--zdim', dest='zdim', help='the dim of z', type=int, default=128)
    parser.add_argument('--num_class', dest='num_class', help='the class number of the data set', type= int, default=5)
    parser.add_argument('--num_image', dest='num_image', help='the image number of the epoch save', type= int, default=64)
    parser.add_argument('--input_size', dest='input_size', help='the input image size of the network', default=128)

    parser.add_argument('--lr', dest='lr', help='The learning rate of the optimizer', type=float, default=2.0e-4)
    parser.add_argument('--beta1', dest='beta1', help='The beta1 param of the Adam optimizer', type=float, default=0.5)
    parser.add_argument('--LAMBDA', dest='LAMBDA', help='the weight of gradient penalty loss', type=float, default=10)
    parser.add_argument('--lambda1', dest='lambda1', help='the weight of KL loss at update E', type=float, default=1)
    parser.add_argument('--lambda2', dest='lambda2', help='the weight of G loss at update E', type=float, default=1)

    args = parser.parse_args()
    return args



def train(dataloader, model, loss_fun, optimizer, args):
    for epoch in tqdm(range(args.epoch), desc='Epoch'):
        model.e.train()
        model.g1.train()
        model.d1.train()
        model.c.train()
        for i, data in enumerate(dataloader):
            x_contour = data[1].cuda()
            label = data[3]
            assert not torch.any(torch.isnan(x_contour)), 'contour data has nan'
            assert not torch.any(torch.isnan(label)), 'label data has nan'
            # to one hot
            label_oh = label_to_one_hot(label.detach().numpy(), args.num_class)
            lable_r = torch.from_numpy(label_oh).float().cuda()

            #  ------------------------------Algorithm--------------------------------------------------------

            # 1. update C ----------------------------------------------------------------------------------------------
            # C_loss #####
            optimizer.c.zero_grad()
            lable_p_contour, _ = model.c(x_contour)
            C_loss = loss_fun.c(lable_r, lable_p_contour)
            # backward -------------------------------------
            C_loss.backward()
            args.logger.log_value('C_loss', C_loss.item())
            optimizer.c.step()

            # 2. update  D1------------------------------------------------------------------------------------------
            optimizer.d1.zero_grad()

            mu, logvar = model.e(x_contour, lable_r)
            z = samper_z(mu, logvar)
            x_f = model.g1(z.detach(), lable_r)

            # D1_loss #####
            D_r = model.d1(x_contour)
            D_f = model.d1(x_f.detach())
            D1_loss = loss_fun.d1(D_r, D_f, model.d1, x_contour, x_f.detach())
            D1_loss.backward()
            print('D1_loss : {}\n'.format(D1_loss.item()))
            args.logger.log_value('D1_loss', D1_loss.item())
            optimizer.d1.step()



            # 3. update  G1----------------------------------------------------------------------------------------------
            optimizer.g1.zero_grad()

            # G1_loss #####
            D_f = model.d1(x_f)
            _, C_f_r = model.c(x_contour)
            _, C_f_f = model.c(x_f)

            G1_loss = loss_fun.g1(D_f, C_f_r, C_f_f, x_contour, x_f)
            print('G1_loss : {}\n'.format(G1_loss.item()))

            G1_loss.backward()
            optimizer.g1.step()

            # 4. update E ----------------------------------------------------------------------------------------------

            # KL loss ####
            optimizer.e.zero_grad()
            KL_loss = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - 1. - logvar, 1))

            x_f = model.g1(z, lable_r)

            D_f = model.d1(x_f)

            _, C_f_r = model.c(x_contour)
            _, C_f_f = model.c(x_f)

            # G_loss #####

            G1_loss = loss_fun.g1(D_f, C_f_r, C_f_f, x_contour, x_f)

            E_loss = args.lambda1 * KL_loss + args.lambda2 * G1_loss
            E_loss.backward()
            args.logger.log_value('E_loss', E_loss.item())
            optimizer.e.step()


        if epoch % 2 == 0: # test g in every 2 epoch
            epoch_save1(model, args, epoch)



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
    model['d1'] = GoodDiscriminator(dim=64, input_size=args.input_size).cuda()
    model['c'] = Classifier(args.num_class).cuda()
    model = EasyDict(model)
    if args.restore:
        sd_e = torch.load('')
        sd_g1 = torch.load('')
        sd_d1 = torch.load('')
        sd_c = torch.load('')
        model.e.load_state_dict(sd_e)
        model.g1.load_state_dict(sd_g1)
        model.d1.load_state_dict(sd_d1)
        model.c.load_state_dict(sd_c)
    else:
        model.g1.apply(wgan_gp_weights_init)
        model.d1.apply(wgan_gp_weights_init)
        model.c.apply(weights_init)

    # build loss function
    loss_fun = {}
    loss_fun['d1'] = DiscriminatorLoss1(LAMBDA=args.LAMBDA)
    loss_fun['g1'] = GeneratorLoss1()
    loss_fun['c'] = ClassifierLoss()
    loss_fun = EasyDict(loss_fun)

    # build optimizer
    optimizer = {}
    # add the filter if you don't want optimize the parameter have no grads
    optimizer['e'] = optim.Adam(filter(lambda p: p.requires_grad, model.e.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer['g1'] = optim.Adam(model.g1.parameters(), lr=1e-4, betas=(0., 0.999))
    optimizer['d1'] = optim.Adam(model.d1.parameters(), lr=1e-4, betas=(0., 0.999))
    optimizer['c'] = optim.Adam(model.c.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer = EasyDict(optimizer)

    train(dataloader, model, loss_fun, optimizer, args)


if __name__ == '__main__':
    main()