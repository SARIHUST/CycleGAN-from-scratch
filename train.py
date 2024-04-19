'''
This script defines the main training procedure.
'''

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.dataset import A2BDataset
from util.util import save_checkpoint, load_checkpoint, save_images, set_seed
import config
from models.cyclegan_model import CycleGANModel
from tqdm import tqdm


if __name__ == '__main__':
    set_seed(423)
    
    train_round = 0     # global training rounds, used for tensorboard writer
    
    # define the CycleGAN model
    cyc_gan = CycleGANModel(isTrain=True)
    
    # define the dataset and data loader
    train_dataset = A2BDataset(root_A=config.TRAIN_DIR_A, root_B=config.TRAIN_DIR_B, transform=config.transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    writer = SummaryWriter('log')

    if config.LOAD_MODEL:
        checkpoint = load_checkpoint(config.CHECKPOINT_FILE_PATH)
        cyc_gan.load_state_dict(checkpoint)

    for epoch in range(config.NUM_EPOCHS):
        real_A, fake_A, real_B, fake_B = 0, 0, 0, 0
        loop = tqdm(train_loader, leave=True)
        for idx, (img_a, img_b) in enumerate(loop):
            img_a = img_a.to(config.DEVICE)
            img_b = img_b.to(config.DEVICE)
            cyc_gan.set_input(img_a, img_b)
            cyc_gan.optimize_parameters()

            # get the training statistics
            cur_losses = cyc_gan.get_current_losses()
            writer.add_scalars('Generator and Discriminator Losses', cur_losses, train_round)
            cur_confidence = cyc_gan.get_current_confidence()
            writer.add_scalars('Discriminators\' prediction confidence', cur_confidence, train_round)
            real_A += cur_confidence['conf_real_A'].item()
            fake_A += cur_confidence['conf_fake_A'].item()
            real_B += cur_confidence['conf_real_B'].item()
            fake_B += cur_confidence['conf_fake_B'].item()

            if (idx + 1) % config.SAVE_ROUND == 0:
                images_A, images_B = cyc_gan.get_current_images()
                save_images(images_A, images_B, writer, idx, train_round)

                if (idx + 1) % (2 * config.SAVE_ROUND) == 0 and config.SAVE_MODEL:
                    checkpoint = cyc_gan.state_dict()
                    save_checkpoint(config.CHECKPOINT_FILE_PATH, checkpoint)
                    # if the training time is too long for an epoch, this will reduce the redundant training time 
                    # if the program is disconnected from Google Colab

            loop.set_description('Epoch {}'.format(epoch + 1))
            loop.set_postfix(
                real_A=real_A / (idx + 1),
                fake_A=fake_A / (idx + 1),
                real_B=real_B / (idx + 1),
                fake_B=fake_B / (idx + 1)
            )

            train_round += 1

        if config.SAVE_MODEL:
            checkpoint = cyc_gan.state_dict()
            save_checkpoint(config.CHECKPOINT_FILE_PATH, checkpoint, epoch + 1)

    writer.close()