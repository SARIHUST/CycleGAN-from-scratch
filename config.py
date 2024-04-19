'''
This script defines some basic CycleGAN parameters, and is modified from 
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/CycleGAN/config.py
'''

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DIR_A = 'data/horse2zebra/train/horses'
TRAIN_DIR_B = 'data/horse2zebra/train/zebras'
VAL_DIR_A = 'data/horse2zebra/validation/horses'
VAL_DIR_B = 'data/horse2zebra/validation/zebras'
BATCH_SIZE = 1
POOL_SIZE = 50
SAVE_ROUND = 400
# mind that the training procedure and networks are based on the one-image per batch legacy, 
# do not change the batch size if you don't wish to modify the other scripts
LEARNING_RATE = 2e-5
BETAS = (0.5, 0.99)
LAMBDA_IDENTITY = 0.0               # controls the identity loss ratio for G_A(x_A) ≈ x_A
LAMBDA_CYCLE_A = 10                 # controls the cycle loss ratio for G_A(G_B(x_A)) ≈ x_A
LAMBDA_CYCLE_B = 10                 # controls the cycle loss ratio for G_B(G_A(x_B)) ≈ x_B
NUM_WORKERS = 1
NUM_EPOCHS = 10
LOAD_MODEL = False                   # set to True if you wish to load pretrained models
SAVE_MODEL = True                   # set to True if you wish to save the trained models
CHECKPOINT_FILE_PATH = 'CycleGAN.pt.tar'

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={'image0': 'image'},
)