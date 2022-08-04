import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F


class ThresholdEncoder(nn.Module):
    def __init__(self, c=8, ratio=8):
        super(ThresholdEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*ratio, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(c*ratio, affine=True),
            nn.ReLU(inplace=True)
        )

        self.threshold_logvar = nn.Conv2d(in_channels=ratio*c, out_channels=1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=True)
        self.threshold_mean = nn.Conv2d(in_channels=ratio*c, out_channels=1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=True)

    def forward(self, x):
        y = self.network(x)
        y = torch.mean(y, dim=-1, keepdim=True)
        y = torch.mean(y, dim=-2, keepdim=True)
        return self.threshold_mean(y), self.threshold_logvar(y)


class ThresholdDecoder(nn.Module):
    def __init__(self, c=8, ratio=8):
        super(ThresholdDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*ratio, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(c*ratio, affine=True),
            nn.ReLU(inplace=True)
        )

        self.threshold_logvar = nn.Conv2d(in_channels=ratio*c, out_channels=1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=True)
        self.threshold_mean = nn.Conv2d(in_channels=ratio*c, out_channels=1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=True)

    def forward(self, x):
        y = self.network(x)
        return self.threshold_mean(y), self.threshold_logvar(y)


class UnetBPL(nn.Module):
    """
    """
    def __init__(self, in_ch, width, depth, out_ch, norm='in', ratio=8, detach=True):
        super(UnetBPL, self).__init__()
        if out_ch == 2:
            out_ch = 1
        self.detach_bpl = detach
        self.segmentor = Unet(in_ch, width, depth, out_ch, norm=norm, side_output=True)
        self.encoder = ThresholdEncoder(c=width, ratio=ratio)
        self.decoder = ThresholdDecoder(c=width*ratio, ratio=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        output, threshold_input = self.segmentor(x)
        if self.detach_bpl is True:
            mu, logvar = self.encoder(threshold_input.detach())
        else:
            mu, logvar = self.encoder(threshold_input)

        z = self.reparameterize(mu, logvar)

        t_mu, t_logvar = self.decoder(z)

        return {'segmentation':output,
                'mu': mu,
                'logvar': logvar,
                'threshold mu': t_mu,
                'threshold var': t_logvar}


class Unet(nn.Module):
    """
    U-net: any depth and any width
    Depth: at least
    """
    def __init__(self, in_ch, width, depth, classes, norm='in', side_output=False):
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(Unet, self).__init__()

        assert depth > 1
        self.depth = depth

        if classes == 2:
            classes = 1

        self.side_output_mode = side_output

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

        output = self.conv_last(y)
        if self.side_output_mode is False:
            return output
        else:
            return output, y


def double_conv(in_channels, out_channels, step, norm):
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===========================================
    if norm == 'in':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'bn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'ln':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'gn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU()
        )