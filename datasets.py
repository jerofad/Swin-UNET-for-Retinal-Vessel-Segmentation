### Import statements 
import os
import numpy as np
import random
import cv2
import PIL.Image
from PIL.Image import open
import torch
from torch.utils.data import Dataset
from glob import glob


""" Dataset Class for Fundus Image Segmentation. Dataset Implememnted includes:
    FIVES Dataset:
    DRIVE Dataset:
    STARE Dataset:
    CHASEDB Dataset:
"""

class FundusDataset(Dataset):
    """ This wworks for DRIVE and FIVES"""
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224,224))
        image = image / 255.0  ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224,224))
        mask = mask / 255.0  ## (512, 512)
        mask = np.expand_dims(mask, axis=0)  ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)


        sample = {'image': image, 'label': mask}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.images_path[index]

        return sample

    def __len__(self):
        return self.n_samples



def load_Fives_images():

    train_x = sorted(glob("datasets/FIVES/train/Original/*"))[:-1] #remove the last db file
    train_y = sorted(glob("datasets/FIVES/train/Ground truth/*"))

    valid_x = sorted(glob("datasets/FIVES/test/Original/*"))
    valid_y = sorted(glob("datasets/FIVES/test/Ground truth/*"))

    return train_x, train_y, valid_x, valid_y


# def get_loader(config):

#     def worker_init_fn(worker_id):
#         random.seed(config['seed'].seed + worker_id)


#     train_x, train_y, valid_x, valid_y = load_Fives_images()

#     # TODO:Transforms 
#     train_dataset = FundusDataset(train_x, train_y)
#     val_dataset = FundusDataset(valid_x, valid_y)


#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
#                                                drop_last=True, pin_memory=True, num_workers=config['num_workers'])
    
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True,
#                                              num_workers=config['num_workers'], pin_memory=True,
#                                              worker_init_fn=worker_init_fn)


#     print(f' Train size: {len(train_dataset)},\n'
#           f' Validation size: {len(val_dataset)}')

#     return train_loader, val_loader