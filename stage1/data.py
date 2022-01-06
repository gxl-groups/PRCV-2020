import os
import torch

from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import trans_To255


class Tradesy(Dataset):
    def __init__(self, sketch_path, contour_path, mask_path,
                 transforms_sketch=None, transforms_contour=None, transforms_mask=None):
        self.classList = os.listdir(sketch_path)
        self.classList.sort()
        self.sketch_path_list = [os.path.join(path, file_name)
                              for path, _, file_list in os.walk(sketch_path)
                              for file_name in file_list if file_name.endswith('.jpg')]
        self.contour_path_list = [os.path.join(path, file_name)
                                 for path, _, file_list in os.walk(contour_path)
                                 for file_name in file_list if file_name.endswith('.jpg')]
        self.mask_path_list = [os.path.join(path, file_name)
                                  for path, _, file_list in os.walk(mask_path)
                                  for file_name in file_list if file_name.endswith('.jpg')]

        self.transforms_sketch = transforms_sketch
        self.transforms_contour = transforms_contour
        self.transforms_mask = transforms_mask

    def __len__(self):
        return len(self.sketch_path_list)

    def __getitem__(self, item):

        label = self.classList.index(self.sketch_path_list[item].split('/')[-2])
        img_sketch = Image.open(self.sketch_path_list[item])
        img_contour = Image.open(self.contour_path_list[item])
        mask = Image.open(self.mask_path_list[item])


        if self.transforms_sketch:
            img_sketch = self.transforms_sketch(img_sketch)
        if self.transforms_contour:
            img_contour = self.transforms_contour(img_contour)
        if self.transforms_mask:
            mask = self.transforms_mask(mask)

        # BW
        img_sketch = torch.where(img_sketch>0.5, torch.Tensor([1.]), torch.Tensor([-1.]))
        img_contour = torch.where(img_contour>0.5, torch.Tensor([1.]), torch.Tensor([-1.]))
        mask = torch.where(mask>0.5, torch.Tensor([1.]), torch.Tensor([0.]))

        return img_sketch, img_contour, mask, label


if __name__ == '__main__':
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
    dataset = Tradesy('/data/kmaeii/dataset/tradesy/tradesy_expand/train/sketch',
                      '/data/kmaeii/dataset/tradesy/tradesy_expand/train/contour',
                      '/data/kmaeii/dataset/tradesy/tradesy_expand/train/mask_erosion',
                      transforms_sketch, transforms_contour, transforms_mask)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=6)
    for i, data in enumerate(dataloader):
        img_sketchs = data[0]
        img_contours = data[1]
        img_mask = data[2]

        save_image(img_sketchs.data, '{}/sketch_examples_{}.jpg'.format('.', i), pad_value=0.)
        save_image(img_contours.data, '{}/contour_examples_{}.jpg'.format('.',i), pad_value=0.)
        save_image(img_mask.data, '{}/mask_examples_{}.jpg'.format('.',i), pad_value=0.)

        break