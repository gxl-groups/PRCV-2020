import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import torchsnooper

class Encoder(nn.Module):
    def __init__(self, vgg_path, zdim, num_class):
        super(Encoder,self).__init__()
        self.vgg_path = vgg_path
        self.vgg_feature = self.get_vgg16()
        self.fc1 = nn.Sequential(
            nn.Linear((512*4*4+num_class), 1024),
            nn.BatchNorm1d(1024)
        )
        self.activation1 = nn.LeakyReLU(0.2)
        self.fc_mu = nn.Linear(1024, zdim)
        self.fc_log_var = nn.Linear(1024, zdim)

        # initial the layer which need train
        self.fc1.apply(self.weights_init)
        self.fc_mu.apply(self.weights_init)
        self.fc_log_var.apply(self.weights_init)

    def get_vgg16(self):  # just use the feature layer
        vgg = models.vgg16_bn()
        pre_param = torch.load(self.vgg_path)
        vgg.load_state_dict(pre_param)
        layers = vgg.features
        for i in layers.parameters():
            i.requires_grad = False

        return layers

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0.0)

    def forward(self, x, label):
        bs = x.size(0)
        x = torch.cat((x,x,x), 1)  # make sure the image can use vgg16
        x = self.vgg_feature(x)  # (bs, 512, 4, 4)
        x = x.view(bs, -1)
        x = torch.cat((x, label), 1)
        x = self.fc1(x)
        x = self.activation1(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2)
        )  # 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2)
        )  # 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )  # 16
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )  # 8
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )  # 4

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, num_class),
            # here can't use softmax/sigmoid because the loss is CrossEntorpyLoss
        )

    def forward(self, x):
        bs = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        f = x
        x = x.view(bs,-1)
        x = self.fc(x)

        return x, f

