import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
# from data_tr_3 import trans_To255, Tradesy, DataLoader
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import utils


class Encoder(nn.Module):
    def __init__(self, inc=1, nef=64):
        super(Encoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc, nef, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU()
        )   # 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(nef, nef, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU()
        )   # 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(nef, 2*nef, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.ReLU()
        )   # 16
        self.conv4 = nn.Sequential(
            nn.Conv2d( 2*nef, 2*nef, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.ReLU()
        )   # 8
        self.conv5 = nn.Sequential(
            nn.Conv2d(2*nef, 4*nef, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.ReLU()
        )   # 4
        self.conv6 = nn.Sequential(
            nn.Conv2d(4*nef, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )   # 2

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.conv6(x)
        return out

class Decoder(nn.Module):
    def __init__(self, inc=128, ndf=64):
        super(Decoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(inc, ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU()
        )   # 64
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(ndf, ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU()
        )   # 32
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d( ndf, 2*ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*ndf),
            nn.ReLU()
        )   # 16
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d( 2*ndf, 2*ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*ndf),
            nn.ReLU()
        )   # 8
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(2*ndf, ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU()
        )   # 4
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(ndf, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )   # 2

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.conv6(x)
        return out



# train classifier and save model to classifier_model dir
# if __name__ == '__main__':
#     # args
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     total_epoch = 200
#     out_dir = './edcoder_model_64'
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#
#     encoder = Encoder().cuda()
#     decoder = Decoder().cuda()
#
#     # get dataset
#     transforms_sketch = transforms.Compose([
#         transforms.Resize(128),
#         trans_To255(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#
#     transforms_contour = transforms.Compose([
#         transforms.Resize(128),
#         trans_To255(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#
#     transforms_mask = transforms.Compose([
#         transforms.Resize(128),
#         trans_To255(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     dataset = Tradesy('/data/kmaeii/dataset/tradesy/tradesy_e2/train/sketch',
#                       '/data/kmaeii/dataset/tradesy/tradesy_e2/train/contour',
#                       '/data/kmaeii/dataset/tradesy/tradesy_e2/train/mask_erosion',
#                       transforms_sketch, transforms_contour, transforms_mask)
#     loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16)
#
#     dataset_test = Tradesy('/data/kmaeii/dataset/tradesy/tradesy_e2/test/sketch',
#                            '/data/kmaeii/dataset/tradesy/tradesy_e2/test/contour',
#                            '/data/kmaeii/dataset/tradesy/tradesy_e2/test/mask_erosion',
#                            transforms_sketch, transforms_contour, transforms_mask)
#     loader_test = DataLoader(dataset_test, batch_size=64, shuffle=True, num_workers=16)
#
#     # optimizer
#     optimizer_e = optim.Adam(encoder.parameters(), lr=1e-4, )
#     optimizer_d = optim.Adam(decoder.parameters(), lr=1e-4, )
#
#     # loss
#     loss_fun = nn.MSELoss()
#
#
#     # test pretrain model
#     with torch.no_grad():
#         dic = torch.load('./edcoder_model_64/encoder_epoch_166.pth')
#         encoder.load_state_dict(dic)
#         dic = torch.load('./edcoder_model_64/decoder_epoch_166.pth')
#         decoder.load_state_dict(dic)
#
#         for data in loader_test:
#             img_contour = data
#             img_contour = img_contour.cuda()
#             out = decoder(encoder(img_contour))
#             utils.save_image(
#                 img_contour,
#                 '{}/img_contour_pretrain.png'.format(out_dir),
#                 nrow=8,
#                 normalize=True,
#                 range=(-1, 1),
#             )
#
#             utils.save_image(
#                 out,
#                 '{}/out_pretrain.png'.format(out_dir),
#                 nrow=8,
#                 normalize=True,
#                 range=(-1, 1),
#             )
#             break
#
#
#     # for epoch in tqdm(range(total_epoch)):
#     #     # train model
#     #     encoder.train()
#     #     decoder.train()
#     #     for data in loader:
#     #         img_contour = data
#     #         img_contour = img_contour.cuda()
#     #         out = decoder(encoder(img_contour))
#     #         loss = loss_fun(out, img_contour)
#     #
#     #         optimizer_e.zero_grad()
#     #         optimizer_d.zero_grad()
#     #         loss.backward()
#     #         optimizer_e.step()
#     #         optimizer_d.step()
#     #
#     #
#     #     #test model
#     #     encoder.eval()
#     #     decoder.eval()
#     #     for data in loader_test:
#     #         img_contour = data
#     #         img_contour = img_contour.cuda()
#     #         out = decoder(encoder(img_contour))
#     #         utils.save_image(
#     #             img_contour,
#     #             '{}/img_contour_epoch{}.png'.format(out_dir, epoch),
#     #             nrow=8,
#     #             normalize=True,
#     #             range=(-1, 1),
#     #         )
#     #
#     #         utils.save_image(
#     #             out,
#     #             '{}/out_epoch{}.png'.format(out_dir, epoch),
#     #             nrow=8,
#     #             normalize=True,
#     #             range=(-1, 1),
#     #         )
#     #         break
#     #
#     #     torch.save(encoder.state_dict(),'{}/encoder_epoch_{}.pth'.format(out_dir, epoch))
#     #     torch.save(decoder.state_dict(),'{}/decoder_epoch_{}.pth'.format(out_dir, epoch))


