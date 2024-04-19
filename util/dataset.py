'''
This script designs the specific datasets used for training the CycleGAN model.
'''

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class A2BDataset(Dataset):
    '''
    Currently includes two main dataset types:
    -- Horse2Zebra
    -- Photo2Painting, paintings include cezanne, monet, ukiyoe, and vangogh

    Example of creating a dataset instance:
    >>> train_dir = 'data/photo2painting/train'
    >>> train_dataset = A2BDataset(train_dir + '/monet', train_dir + '/photo')
    '''
    def __init__(self, root_A, root_B, transform=None):
        super().__init__()
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.A_image_paths = os.listdir(root_A)
        self.B_image_paths = os.listdir(root_B)
        self.A_len = len(self.A_image_paths)
        self.B_len = len(self.B_image_paths)
        self.dataset_len = max(self.A_len, self.B_len)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        A_path = os.path.join(self.root_A, self.A_image_paths[index % self.A_len])
        B_path = os.path.join(self.root_B, self.B_image_paths[index % self.B_len])
        A_img = np.array(Image.open(A_path).convert('RGB'))
        B_img = np.array(Image.open(B_path).convert('RGB'))

        if self.transform:
            augment = self.transform(image=A_img, image0=B_img)
            A_img = augment['image']
            B_img = augment['image0']
        
        return A_img, B_img