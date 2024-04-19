'''
This script defines the utility functions that save the checkpoints
'''

import random
import torch
import os
import numpy as np
import config
import matplotlib.pyplot as plt

def save_images(images_A, images_B, writer, idx, train_round):
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax[0][0].set_title('Real A')
    ax[0][0].imshow((images_A['real_A'] * 0.5 + 0.5).squeeze().detach().cpu().permute(1, 2, 0).float())
    ax[0][0].axis('off')
    ax[0][1].set_title('Generated B')
    ax[0][1].imshow((images_A['fake_B'] * 0.5 + 0.5).squeeze().detach().cpu().permute(1, 2, 0).float())
    ax[0][1].axis('off')
    ax[0][2].set_title('Reconstructed A')
    ax[0][2].imshow((images_A['rec_A'] * 0.5 + 0.5).squeeze().detach().cpu().permute(1, 2, 0).float())
    ax[0][2].axis('off')
    ax[1][0].set_title('Real B')
    ax[1][0].imshow((images_B['real_B'] * 0.5 + 0.5).squeeze().detach().cpu().permute(1, 2, 0).float())
    ax[1][0].axis('off')
    ax[1][1].set_title('Generated A')
    ax[1][1].imshow((images_B['fake_A'] * 0.5 + 0.5).squeeze().detach().cpu().permute(1, 2, 0).float())
    ax[1][1].axis('off')
    ax[1][2].set_title('Reconstructed B')
    ax[1][2].imshow((images_B['rec_B'] * 0.5 + 0.5).squeeze().detach().cpu().permute(1, 2, 0).float())
    ax[1][2].axis('off')
    plt.savefig('saved-figs/real-fake-rec_{}.png'.format(idx + 1))
    writer.add_figure('outputs', fig, train_round + 1)

def save_checkpoint(checkpoint_file, cycle_gan, epoch=None):
    if epoch:
        print('=> saving checkpoint on epoch {}'.format(epoch))
    checkpoint = cycle_gan
    torch.save(checkpoint, checkpoint_file)


def load_checkpoint(checkpoint_file):
    print('=> loading checkpoint')
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    return checkpoint


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False