import torch
import torch.nn as nn
import numpy as np
import torchsnooper
import pysnooper
from utils import calc_gradient_penalty

class ClassifierLoss(nn.Module):
    def __init__(self):
        super(ClassifierLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, c_real, c_predict):
        # one hot to label
        real = torch.topk(c_real, 1)[1].squeeze(1)
        loss = self.criterion(c_predict, real)
        return loss

# ----------------------------------D1 G1 loss -------------------------------------------------------------------------
class DiscriminatorLoss1(nn.Module):
    def __init__(self, LAMBDA):
        super(DiscriminatorLoss1, self).__init__()
        self.LAMBDA = LAMBDA

    # @pysnooper.snoop(prefix='DiscriminatorLoss forward')
    def forward(self,D_r, D_f, netD, real_data, fake_data):
        loss_real = D_r.mean()
        loss_fake = D_f.mean()
        loss_grad_penalty = calc_gradient_penalty(netD,  real_data, fake_data, self.LAMBDA)
        loss_D = loss_fake + loss_grad_penalty - loss_real
        return loss_D

class GeneratorLoss1(nn.Module):
    def __init__(self):
        super(GeneratorLoss1, self).__init__()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_D = nn.BCELoss()

    def cri(self, x_r, x_f):
        mask = torch.eq(x_r, -1).float()
        num = torch.nonzero(mask).shape[0]
        dif = nn.L1Loss(reduction='sum')(x_r * mask, x_f * mask)
        loss = dif / num
        return loss

    def forward(self, D_f, C_f_r, C_f_f, x_r, x_f):
        loss_gan = - D_f.mean() * 0.25
        loss_c = self.criterion_L1(C_f_r, C_f_f) *15
        loss_x = self.criterion_L1(x_r, x_f) * 8
        loss_x2 = self.cri(x_r, x_f) * 15
        loss_g = loss_gan + loss_c + loss_x + loss_x2

        # print('G_loss_gan : {}\n'.format(loss_gan.item()))
        # print('G_loss_c : {}\n'.format(loss_c.item()))
        # print('G_loss_x : {}\n'.format(loss_x.item()))
        # print('G_loss_x2 : {}\n'.format(loss_x2.item()))
        return  loss_g

# ----------------------------------D2 G2 loss -------------------------------------------------------------------------
class DiscriminatorLoss2(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss2, self).__init__()
        self.criterion = nn.MSELoss()


    # @pysnooper.snoop(prefix='DiscriminatorLoss forward')
    def forward(self,D_r, D_f):
        y_pos = torch.ones_like(D_r)
        y_neg = torch.zeros_like(D_r)
        loss_r = self.criterion(D_r, y_pos)
        loss_f = self.criterion(D_f, y_neg)
        return loss_r + loss_f

class GeneratorLoss2(nn.Module):
    def __init__(self, args):
        super(GeneratorLoss2, self).__init__()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_D = nn.MSELoss()
        self.args = args


    def cri(self, x_r, x_f):
        mask = torch.eq(x_r, -1).float()
        num = torch.nonzero(mask).shape[0]
        dif = nn.L1Loss(reduction='sum')(x_r * mask, x_f * mask)
        loss = dif / num
        return loss

    def cri_feature(self, D_out_r, D_out_f):
        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (3 + 1)   # feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        for i in range(len(D_out_r) - 1):
            loss_G_GAN_Feat += feat_weights * self.criterion_L1(D_out_f[i], D_out_r[i].detach())

        return loss_G_GAN_Feat


    def forward(self, D_out_r, D_out_f, x_r, x_f):
        D_f = D_out_f[-1]
        y_pos = torch.ones_like(D_f)
        loss_gan = self.criterion_D(D_f, y_pos)
        loss_x = self.criterion_L1(x_r, x_f) * 15
        loss_f = self.cri_feature(D_out_r, D_out_f) * 10

        loss_g = loss_gan + loss_x + loss_f
        print('G2_loss_gan : {}\n'.format(loss_gan.item()))
        print('G2_loss_x : {}\n'.format(loss_x.item()))
        print('G2_loss_f : {}\n'.format(loss_f.item()))
        self.args.logger.log_value('G2_loss_gan', loss_gan.item())
        self.args.logger.log_value('G2_loss_x', loss_x.item())
        self.args.logger.log_value('G2_loss_f', loss_f.item())
        return loss_g

# ----------------------------------D3 G3 loss -------------------------------------------------------------------------
class DiscriminatorLoss3(nn.Module):
    def __init__(self, LAMBDA):
        super(DiscriminatorLoss3, self).__init__()
        self.LAMBDA = LAMBDA

    # @pysnooper.snoop(prefix='DiscriminatorLoss forward')
    def forward(self,D_r, D_f, netD, real_data, fake_data):
        loss_real = D_r.mean()
        loss_fake = D_f.mean()
        loss_grad_penalty = calc_gradient_penalty(netD,  real_data, fake_data, self.LAMBDA)
        loss_D = loss_fake + loss_grad_penalty - loss_real
        return loss_D

class GeneratorLoss3(nn.Module):
    def __init__(self, model_feat, args):
        super(GeneratorLoss3, self).__init__()
        self.model_feat = model_feat
        self.criterion_D = nn.MSELoss()
        self.args = args

    def criterion_perceptual(self, x_r, x_f):
        x_r = torch.cat((x_r,x_r,x_r), dim=1)
        x_f = torch.cat((x_f,x_f,x_f), dim=1)
        percep_r = self.model_feat(x_r)
        percep_f = self.model_feat(x_f)
        loss = torch.tensor(0.).cuda()
        for i in range(len(percep_r)):
            loss += torch.nn.MSELoss()(percep_r[i], percep_f[i])
        return loss

    def forward(self, D_f, x_r, x_f):
        loss_gan = - D_f.mean()
        # loss_mx = self.criterion_mx(x_r, x_f)
        # loss_per = self.criterion_perceptual(x_r, x_f)
        loss_g = loss_gan

        print('G2_loss_gan : {}\n'.format(loss_gan.item()))
        # print('G2_loss_mx : {}\n'.format(loss_mx.item()))
        # print('G2_loss_per : {}\n'.format(loss_per.item()))
        self.args.logger.log_value('G2_lgan', loss_gan.item())
        # self.args.logger.log_value('G2_lmx', loss_mx.item())
        # self.args.logger.log_value('G2_lper', loss_per.item())
        return loss_g