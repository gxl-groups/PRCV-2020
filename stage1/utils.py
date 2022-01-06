import torch.nn.init as init
import torch.nn as nn
import torch
import numpy as np
import os
from torchvision.utils import save_image
from wgan_gp import MyConvo2d
from torch import autograd
from PIL import Image

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


def wgan_gp_weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def pix_init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA):
    BATCH_SIZE = real_data.size(0)
    DIM = real_data.size(2)
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 1, DIM, DIM)
    alpha = alpha.cuda()

    # fake_data = fake_data.view(BATCH_SIZE, 1, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def samper_z( mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    return eps.mul(std).add_(mu)


def print_gradint(net):
    params = list(net.named_parameters())
    for grad_wt in range(len(params)):
        norm_v = torch.norm(params[grad_wt][1].grad).cpu().data.numpy() if params[grad_wt][1].grad is not None else 0

        print('Param %-3s Name %-80s Grad_Norm %-20s'%
              (str(grad_wt),
               params[grad_wt][0],
               str(norm_v)))


def label_to_one_hot(labels, num_classes):
    """Convert class labels from scalars to one-hot vectors.
       labels is a list  or numpy.array like [1,3,5] -----> size=(n,)
    """

    one_hot_label = np.eye(num_classes)[labels]
    return one_hot_label

def epoch_save1(model_dic, args, num_epoch):
    net = model_dic.g1
    net.eval()
    with torch.no_grad():
        z_p = torch.randn(args.num_image, args.zdim).cuda()

        c_p = torch.FloatTensor(args.num_image, args.num_class)
        c_p.zero_()
        rand_label = torch.LongTensor(args.num_image, 1).random_() % args.num_class
        c_p = c_p.scatter_(1, rand_label, 1).cuda()

        image1 = net(z_p, c_p)
        save_image(image1.data.cpu(), '{}/epoch_{}_save_1.jpg'.format(args.output_dir, num_epoch+1), pad_value=0.)


        if num_epoch % 4 == 0:
            model_save_path = args.output_dir + '/model_save'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            torch.save(model_dic.g1.state_dict(), '{}/G1_epoch_{}.pth'.format(model_save_path, num_epoch+1))
            torch.save(model_dic.d1.state_dict(), '{}/D1_epoch_{}.pth'.format(model_save_path, num_epoch+1))
            torch.save(model_dic.c.state_dict(), '{}/C_epoch_{}.pth'.format(model_save_path, num_epoch+1))
            torch.save(model_dic.e.state_dict(), '{}/E_epoch_{}.pth'.format(model_save_path, num_epoch+1))


def epoch_save2(model_dic, args, num_epoch):
    net1 = model_dic.g1
    net2 = model_dic.g2
    net2.eval()
    with torch.no_grad():
        z_p = torch.randn(args.num_image, args.zdim).cuda()

        c_p = torch.FloatTensor(args.num_image, args.num_class)
        c_p.zero_()
        rand_label = torch.LongTensor(args.num_image, 1).random_() % args.num_class
        c_p = c_p.scatter_(1, rand_label, 1).cuda()

        image1 = net1(z_p, c_p)
        image2 = net2(image1)
        save_image(image1.data.cpu(), '{}/epoch_{}_save_1.jpg'.format(args.output_dir, num_epoch+1), pad_value=0.)
        save_image(image2.data.cpu(), '{}/epoch_{}_save_2.jpg'.format(args.output_dir, num_epoch+1), pad_value=0.)


        if num_epoch % 6 == 0:
            model_save_path = args.output_dir + '/model_save'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            torch.save(model_dic.g2.state_dict(), '{}/G2_epoch_{}.pth'.format(model_save_path, num_epoch+1))
            torch.save(model_dic.d2.state_dict(), '{}/D2_epoch_{}.pth'.format(model_save_path, num_epoch+1))

def epoch_save3(model_dic, args, num_epoch):
    net1 = model_dic.g1
    net2 = model_dic.g2
    net3 = model_dic.g3
    net3.eval()
    with torch.no_grad():
        z_p = torch.randn(args.num_image, args.zdim).cuda()

        c_p = torch.FloatTensor(args.num_image, args.num_class)
        c_p.zero_()
        rand_label = torch.LongTensor(args.num_image, 1).random_() % args.num_class
        c_p = c_p.scatter_(1, rand_label, 1).cuda()

        image1 = net1(z_p, c_p)
        image2 = net2(image1)
        image3 = net3(image2)
        save_image(image1.data.cpu(), '{}/epoch_{}_save_1.jpg'.format(args.output_dir, num_epoch+1), pad_value=0.)
        save_image(image2.data.cpu(), '{}/epoch_{}_save_2.jpg'.format(args.output_dir, num_epoch+1), pad_value=0.)
        save_image(image3.data.cpu(), '{}/epoch_{}_save_3.jpg'.format(args.output_dir, num_epoch+1), pad_value=0.)


        if num_epoch % 8 == 0:
            model_save_path = args.output_dir + '/model_save'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            torch.save(model_dic.g3.state_dict(), '{}/G3_epoch_{}.pth'.format(model_save_path, num_epoch+1))
            torch.save(model_dic.d3.state_dict(), '{}/D3_epoch_{}.pth'.format(model_save_path, num_epoch+1))

class FeatureExtractor(nn.Module):
    # Extract features from intermediate layers of a network

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]

class trans_To255(object):

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic_np = np.array(pic)
        dim = len(pic_np.shape)
        if dim == 3:
            for i in range(3):
                pic_np[:, :, i] = np.absolute(
                    (pic_np[:, :, i] - np.min(pic_np[:, :, i])) / (np.max(pic_np[:, :, i]) - np.min(pic_np[:, :, i]))) * 255
            ret = Image.fromarray(pic_np)
        elif dim == 2:
            pic_np = np.absolute((pic_np - np.min(pic_np)) / (np.max(pic_np) - np.min(pic_np))) * 255
            ret = Image.fromarray(pic_np).convert('L')

        return ret