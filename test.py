from PIL import Image
import torchvision
from util.util import load_checkpoint
import config
import numpy as np
from models.cyclegan_model import CycleGANModel
from torchvision import transforms as trans
import os

# cyc_gan = CycleGANModel(isTrain=False)
# cyc_gan.load_state_dict(load_checkpoint(config.CHECKPOINT_FILE_PATH))

# for img_name in os.listdir('test-input'):
#     test_img = np.array(Image.open('test-input/' + img_name).convert('RGB'))
#     h, w = test_img.shape[0], test_img.shape[1]
#     print(test_img.shape)
#     aug = config.transforms(image=test_img, image0=test_img)
#     test_img = aug['image'].to(config.DEVICE)

#     fake = cyc_gan.netG_B(test_img).detach().cpu()
#     change_back_size = trans.Resize((h, w))
#     fake = change_back_size(fake * 0.5 + 0.5)
#     print(fake.shape)
#     torchvision.utils.save_image(fake, 'test-output/CycleGAN/' + img_name)

# img_name = 'person3.jpg'
# test_img = np.array(Image.open('test-input/' + img_name).convert('RGB'))
# h, w = test_img.shape[0], test_img.shape[1]
# print(test_img.shape)
# aug = config.transforms(image=test_img, image0=test_img)
# test_img = aug['image'].to(config.DEVICE)
# fake = cyc_gan.netG_B(test_img).detach().cpu()
# change_back_size = trans.Resize((h, w))
# fake = change_back_size(fake * 0.5 + 0.5)
# print(fake.shape)
# torchvision.utils.save_image(fake, 'test-output/CycleGAN-1' + img_name)

# exit()

for i in range(3):
    print('=============== testing on model {} ==============='.format(i))

    cyc_gan = CycleGANModel(isTrain=False)
    cyc_gan.load_state_dict(load_checkpoint('{}_CycleGAN.pt.tar'.format(i)))

    for img_name in os.listdir('test-input'):
        test_img = np.array(Image.open('test-input/' + img_name).convert('RGB'))
        h, w = test_img.shape[0], test_img.shape[1]
        print(test_img.shape)
        aug = config.transforms(image=test_img, image0=test_img)
        test_img = aug['image'].to(config.DEVICE)

        fake = cyc_gan.netG_B(test_img).detach().cpu()
        change_back_size = trans.Resize((h, w))
        fake = change_back_size(fake * 0.5 + 0.5)
        print(fake.shape)
        torchvision.utils.save_image(fake, 'test-output/CycleGAN-{}/'.format(i) + img_name)