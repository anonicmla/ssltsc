import os
import pdb
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from dl4d.images import ImageDataset


class Suresort(ImageDataset):
    # folder from suresort parent level
    base_folder = 'preprocessed/Datenlieferung_1/rawdata/selection/model_data'
    seed = 1337

    def __init__(self, root, part='train', task='classification',
                 features=False,
                 val_size=None,
                 test_size=None,
                 transform=None, target_transform=None, download=True,
                 normalize=False, standardize=False,
                 scale_overall=True, scale_channelwise=True):

        self.root = root
        if download:
            self.download()

        super(Suresort, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.x, self.y = self.load_dataset(part=part)

    def __len__(self):
        return len(self.x)

    def download(self):
        pass