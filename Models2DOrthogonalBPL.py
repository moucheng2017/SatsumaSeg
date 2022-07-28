import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

# ==================
# Blocks
# ==================


def double_conv(in_channels, out_channels, step):
    # double convolutional layers
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), stride=step, padding=(1, 1), groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, (3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.PReLU()
    )


def single_conv(in_channels, out_channels, step):
    # single convolutional layers
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (7, 7), stride=step, padding=(3, 3), groups=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.PReLU()
    )


class DoubleRandomDilatedConv(nn.Module):
    # Random dilation convolutional layers
    def __init__(self, in_channels, out_channels, step):
        super(DoubleRandomDilatedConv, self).__init__()
        self.attention_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=step, dilation=(1, 1), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU()
        )

    def forward(self, x, random_seed):
        self.attention_branch[0].dilation = (int(random_seed), int(random_seed))
        self.attention_branch[0].padding = (int(random_seed), int(random_seed))
        self.attention_branch[3].dilation = (int(random_seed), int(random_seed))
        self.attention_branch[3].padding = (int(random_seed), int(random_seed))
        output = self.attention_branch(x)
        return output


class ThresholdModel2D(nn.Module):
    def __init__(self, c=8):
        super(ThresholdModel2D, self).__init__()

        self.threshold_net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*8, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(c*8, affine=True),
            nn.ReLU(inplace=True)
        )

        self.threshold_logvar = nn.Conv2d(in_channels=8*c, out_channels=1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=True)

        self.threshold_mean = nn.Conv2d(in_channels=8 * c, out_channels=1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=True)

    def forward(self, x):
        y = self.threshold_net(x.detach())
        y = torch.mean(y, dim=-1, keepdim=True)
        y = torch.mean(y, dim=-2, keepdim=True)
        t_mean = self.threshold_mean(y)
        t_logvar = self.threshold_logvar(y)
        return t_mean, t_logvar


class cm_net(nn.Module):
    """ This class defines the confusion matrix network
    """
    def __init__(self, c, h, w, class_no=2, latent=512):
        super(cm_net, self).__init__()
        self.fc_encoder = nn.Linear(c * h * w, latent)
        self.fc_decoder = nn.Linear(latent, h * w * class_no ** 2)
        self.act = nn.ReLU(inplace=True) # relu is better than prelu in mnist

    def forward(self, x):
        cm = torch.flatten(x, start_dim=1)
        cm = self.fc_encoder(cm)
        cm = self.act(cm)
        cm = self.fc_decoder(cm)
        cm = F.softplus(cm)
        return cm


class Unet(nn.Module):
    """
    U-net
    """
    def __init__(self, in_ch, width, depth, classes, norm='in'):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(Unet, self).__init__()
        self.depth = depth

        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()

        for i in range(self.depth):

            if i == 0:

                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))

            elif i < (self.depth - 1):

                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))

            else:

                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, classes, 1, bias=True)

    def forward(self, x):

        y = x
        encoder_features = []

        for i in range(len(self.encoders)):

            y = self.encoders[i](y)
            encoder_features.append(y)

        for i in range(len(encoder_features)):

            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]

            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])

                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            y = torch.cat([y_e, y], dim=1)
            y = self.decoders[-(i+1)](y)

        y = self.conv_last(y)
        return y