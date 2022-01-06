import torch
import argparse

from wgan_gp import  GoodGenerator
from torchvision.utils import save_image
import os
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zdim', dest='zdim', help='the dim of z', type=int, default=128)
    parser.add_argument('--num_img', dest='num_img', type=int, default=64)
    parser.add_argument('--num_class', dest='num_class', type=int, default=5)
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=2)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='./output_s1/model_save/G_epoch_113.pth')
    args = parser.parse_args()
    return args
args = parser_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

g1 = GoodGenerator().cuda()
dict = torch.load(args.model_dir)
g1.load_state_dict(dict)
g1.eval()


with torch.no_grad():
    for label in range(args.num_class):
        path1 = './output_s1/class_{}_test113.jpg'.format(label)
        # sampler z and c
        z_p = torch.randn(args.num_img, args.zdim).cuda()
        c_p = torch.FloatTensor(args.num_img, args.num_class)
        c_p.zero_()
        rand_label = torch.ones(args.num_img, 1).long() * label
        c_p = c_p.scatter_(1, rand_label, 1).cuda()

        out = g1(z_p, c_p)
        save_image(out, path1, normalize=True)


