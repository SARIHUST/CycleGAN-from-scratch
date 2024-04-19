from models.cyclegan_model import CycleGANModel
from models.networks import *
from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter('log_model')
# cyc_gan = CycleGANModel(isTrain=True)
# writer.add_graph(cyc_gan.netG_A)
# writer.add_graph(cyc_gan.netD_A)
# writer.add_graph(cyc_gan.netG_B)
# writer.add_graph(cyc_gan.netD_B)
# g = Generator()
d = Discriminator()
# print(g)
input = torch.randn(1, 3, 256, 256)
# print(g(input).shape)
# writer.add_graph(g, input)
writer.add_graph(d, input)
writer.close()