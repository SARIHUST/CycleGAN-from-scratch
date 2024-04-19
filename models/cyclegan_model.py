import torch
import torch.nn as nn
from . import networks
from util.image_pool import ImagePool
import config

class CycleGANModel():
    def __init__(self, isTrain=True) -> None:
        # get the basic parameters from config
        self.device = config.DEVICE
        self.lambda_cyc_a = config.LAMBDA_CYCLE_A
        self.lambda_cyc_b = config.LAMBDA_CYCLE_B
        self.lambda_idt = config.LAMBDA_IDENTITY
        self.lr = config.LEARNING_RATE
        self.betas = config.BETAS
        self.pool_size = config.POOL_SIZE
        self.isTrain = isTrain

        # specify the names of the images to display/show
        self.visual_names_A = ['real_A', 'fake_B', 'rec_A']
        self.visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if isTrain and self.lambda_idt > 0:
            self.visual_names_A.append('idt_A')
            self.visual_names_B.append('idt_B')

        # specify the name of the losses
        self.loss_names = ['loss_D_A', 'loss_G_A', 'loss_D_B', 'loss_G_B', 'loss_cyc_A', 'loss_cyc_B']
        if isTrain and self.lambda_idt > 0:
            self.loss_names.extend(['loss_idt_A', 'loss_idt_B'])

        # specify the name of the 'confidence' of the discriminator's prediction
        self.conf_names = ['conf_real_A', 'conf_fake_A', 'conf_real_B', 'conf_fake_B']

        # specify the models name
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # define the models
        self.netG_A = networks.Generator().to(self.device)          # generates images of type A
        self.netG_B = networks.Generator().to(self.device)          # generates images of type B
        self.netD_A = networks.Discriminator().to(self.device)      # discriminizes images of type A
        self.netD_B = networks.Discriminator().to(self.device)      # discriminizes images of type B

        if self.isTrain:
            # define the losses
            self.criterionGAN = nn.MSELoss()
            self.criterionCyc = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()

            # define the optimizers
            self.optim_G = torch.optim.Adam(
                list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
                lr=self.lr,
                betas = self.betas
            )
            self.optim_D = torch.optim.Adam(
                list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
                lr=self.lr,
                betas = self.betas
            )

            # define the image pools to store generated images
            self.fake_A_pool = ImagePool(self.pool_size)
            self.fake_B_pool = ImagePool(self.pool_size)

    def set_input(self, img_A, img_B):
        self.real_A = img_A
        self.real_B = img_B
        
    def get_current_losses(self):
        '''Returns the current losses'''
        losses = {}
        for name in self.loss_names:
            losses[name] = float(getattr(self, name))
        return losses

    def get_current_images(self):
        '''Returns the current images'''
        image_map_A, image_map_B = {}, {}
        for name in self.visual_names_A:
            image_map_A[name] = getattr(self, name)
        for name in self.visual_names_B:
            image_map_B[name] = getattr(self, name)
        return image_map_A, image_map_B

    def get_current_confidence(self):
        '''Returns the current confidence'''
        confidence = {}
        for name in self.conf_names:
            confidence[name] = getattr(self, name)
        return confidence

    def forward(self):
        '''Run the forward pass to get the generated and reconstructed images'''
        self.fake_A = self.netG_A(self.real_B)
        self.fake_B = self.netG_B(self.real_A)
        self.rec_A = self.netG_A(self.fake_B)
        self.rec_B = self.netG_B(self.fake_A)

    def backward_D_basis(self, netD, real, fake):
        '''Calculate the GAN loss for a certain discriminator
        Parameters:
            netD (network)      -- the certain discriminator
            real (tensor array) -- real image
            fake (tensor array) -- generated image
        Return the loss of the discriminator, and calls backward to compute the gradients
        '''
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        # combine the loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        return loss_D, pred_real.mean(), pred_fake.mean()

    def backward_D_A(self):
        '''Calculate the GAN loss for the discriminator D_A, and the confidence'''
        fake_A = self.fake_A_pool.query(self.fake_A)    # 50% to return an earlier generated image
        self.loss_D_A, self.conf_real_A, self.conf_fake_A = self.backward_D_basis(self.netD_A, self.real_A, fake_A)

    def backward_D_B(self):
        '''Calculate the GAN loss for the discriminator D_B, and the confidence'''
        fake_B = self.fake_B_pool.query(self.fake_B)    # 50% to return an earlier generated image
        self.loss_D_B, self.conf_real_B, self.conf_fake_B = self.backward_D_basis(self.netD_B, self.real_B, fake_B)

    def backward_G(self):
        '''Calculate the loss for generators G_A and G_B'''
        # identity loss
        if self.lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_A)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A) * self.lambda_idt
            self.idt_B = self.netG_B(self.real_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) * self.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        pred_fake_A = self.netD_A(self.fake_A)
        self.loss_G_A = self.criterionGAN(pred_fake_A, torch.ones_like(pred_fake_A))
        pred_fake_B = self.netD_B(self.fake_B)
        self.loss_G_B = self.criterionGAN(pred_fake_B, torch.ones_like(pred_fake_B))

        # forward cycle loss -- G_A(G_B(real_A)) vs real_A
        self.loss_cyc_A = self.criterionCyc(self.rec_A, self.real_A)
        # backward cycle loss -- G_B(G_A(real_B)) vs real_B
        self.loss_cyc_B = self.criterionCyc(self.rec_B, self.real_B)

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cyc_A + self.loss_cyc_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        '''Generate the images, calculate the loss, gradients, and update the network weights'''
        # forward pass
        self.forward()
        # generators are trained first to create the image pools
        self.optim_G.zero_grad()
        self.backward_G()
        self.optim_G.step()
        # discriminators
        self.optim_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optim_D.step()

    def state_dict(self):
        if self.isTrain:
            total_state_dict = {
                'G_A': self.netG_A.state_dict(),
                'G_B': self.netG_B.state_dict(),
                'D_A': self.netD_A.state_dict(),
                'D_B': self.netD_B.state_dict(),
                'optim_G': self.optim_G.state_dict(),
                'optim_D': self.optim_D.state_dict()
            }
        else:
            total_state_dict = {
                'G_A': self.netG_A.state_dict(),
                'G_B': self.netG_B.state_dict(),
                'D_A': self.netD_A.state_dict(),
                'D_B': self.netD_B.state_dict(),
            }
        return total_state_dict

    def load_state_dict(self, total_state_dict):
        self.netG_A.load_state_dict(total_state_dict['G_A'])
        self.netG_B.load_state_dict(total_state_dict['G_B'])
        self.netD_A.load_state_dict(total_state_dict['D_A'])
        self.netD_B.load_state_dict(total_state_dict['D_B'])
        if self.isTrain:
            self.optim_G.load_state_dict(total_state_dict['optim_G'])
            self.optim_D.load_state_dict(total_state_dict['optim_D'])

    def eval(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.eval()