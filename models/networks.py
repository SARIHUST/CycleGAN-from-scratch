import torch
import torch.nn as nn

class UpDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs) -> None:
        '''
        key arguments include kernel_size, stride, padding and padding_mode
        '''
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs)
            if down
            else
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            UpDownBlock(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            UpDownBlock(channels, channels, use_act=False, kernel_size=3, padding=1, padding_mode='reflect'),
        )

    def forward(self, x):
        return x + self.conv(x)

class Generator(nn.Module):
    def __init__(self, img_channels=3, ngf=64, nrb=9) -> None:
        super().__init__()
        self.initial_head = nn.Sequential(
            nn.Conv2d(img_channels, ngf, 7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        self.down_layers = nn.Sequential(
            UpDownBlock(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            UpDownBlock(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1)
            # UpDownBlock(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            # UpDownBlock(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, padding_mode='reflect')             
        )

        self.residual_layers = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(nrb)]
        )

        self.up_layers = nn.Sequential(
            UpDownBlock(ngf * 4, ngf * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            UpDownBlock(ngf * 2, ngf, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(ngf, img_channels, 7, stride=1, padding=3, padding_mode='reflect'),
            # nn.InstanceNorm2d(ngf),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial_head(x)
        x = self.down_layers(x)
        x = self.residual_layers(x)
        x = self.up_layers(x)
        x = self.final_layer(x)
        return torch.tanh(x)

class InstanceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding=1),
            # nn.Conv2d(in_channels, out_channels, 4, stride, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]) -> None:
        super().__init__()
        self.initial_head = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, stride=2, padding=1),
            # nn.Conv2d(in_channels, features[0], 4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )

        mid_layers = []
        in_channels = features[0]

        for feature in features[1:]:
            mid_layers.append(InstanceBlock(in_channels, feature, stride=2 if feature != features[-1] else 1))
            in_channels = feature
        
        self.mid_layer = nn.Sequential(*mid_layers)

        self.final_layer = nn.Conv2d(in_channels, 1, 4, stride=1, padding=1)
        # self.final_layer = nn.Conv2d(in_channels, 1, 4, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.initial_head(x)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        # return torch.sigmoid(x)
        return x

def test():
    net_D = Discriminator()
    net_G = Generator()
    inputs = torch.randn(1, 3, 256, 256)
    outputs = net_D(inputs)
    print(outputs.shape)

if __name__ == '__main__':
    test()