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
        nn.Conv3d(in_channels, out_channels, (3, 3, 3), stride=step, padding=(0, 1, 1), groups=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.PReLU(),
        nn.Conv3d(out_channels, out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.PReLU()
    )


def single_conv(in_channels, out_channels, step):
    # single convolutional layers
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, (7, 7, 7), stride=step, padding=(3, 3, 3), groups=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.PReLU()
    )


class DoubleRandomDilatedConv(nn.Module):
    # Random dilation convolutional layers
    def __init__(self, in_channels, out_channels, step):
        super(DoubleRandomDilatedConv, self).__init__()
        self.attention_branch = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=step, dilation=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU()
        )

    def forward(self, x, random_seed):
        self.attention_branch[0].dilation = (1, int(random_seed), int(random_seed))
        self.attention_branch[0].padding = (1, int(random_seed), int(random_seed))
        self.attention_branch[3].dilation = (1, int(random_seed), int(random_seed))
        self.attention_branch[3].padding = (1, int(random_seed), int(random_seed))
        output = self.attention_branch(x)
        return output


class ConfModel(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4):
        super(ConfModel, self).__init__()
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.attention_branch = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels*ratio, kernel_size=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.InstanceNorm3d(in_channels*ratio, affine=True),
            nn.PReLU(),
            nn.Conv3d(in_channels=in_channels*ratio, out_channels=out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU()
        )

    def forward(self, x):
        output = self.attention_branch(x)
        output = self.avg(output)
        output = torch.sigmoid(output)
        return output


class Unet3D(nn.Module):
    def __init__(self, in_ch, width, class_no, z_downsample=0):
        super(Unet3D, self).__init__()
        if class_no == 2:
            self.final_in = 1
        else:
            self.final_in = class_no

        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8

        if z_downsample == 0:
            encoder_downsamples = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
            upsamples_steps = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
        elif z_downsample == 1:
            encoder_downsamples = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
            upsamples_steps = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
        elif z_downsample == 2:
            encoder_downsamples = [(2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)]
            upsamples_steps = [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)]
        elif z_downsample == 3:
            encoder_downsamples = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)]
            upsamples_steps = [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        elif z_downsample == 4:
            encoder_downsamples = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
            upsamples_steps = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        else:
            encoder_downsamples = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
            upsamples_steps = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]

        # if z_downsample == 0:
        #     encoder_downsamples = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
        #     upsamples_steps = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
        # elif z_downsample == 1:
        #     encoder_downsamples = [(2, 2, 2), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
        #     upsamples_steps = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (2, 2, 2)]
        # elif z_downsample == 2:
        #     encoder_downsamples = [(2, 2, 2), (2, 2, 2), (1, 1, 1), (1, 1, 1)]
        #     upsamples_steps = [(1, 1, 1), (1, 1, 1), (2, 2, 2), (2, 2, 2)]
        # elif z_downsample == 3:
        #     encoder_downsamples = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)]
        #     upsamples_steps = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        # elif z_downsample == 4:
        #     encoder_downsamples = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        #     upsamples_steps = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        # else:
        #     encoder_downsamples = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
        #     upsamples_steps = [(1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]

        self.econv0 = single_conv(in_channels=in_ch, out_channels=self.w1, step=1)
        self.econv1 = DoubleRandomDilatedConv(in_channels=self.w1, out_channels=self.w2, step=encoder_downsamples[0])
        self.econv2 = DoubleRandomDilatedConv(in_channels=self.w2, out_channels=self.w3, step=encoder_downsamples[1])
        self.econv3 = DoubleRandomDilatedConv(in_channels=self.w3, out_channels=self.w4, step=encoder_downsamples[2])
        self.bridge = DoubleRandomDilatedConv(in_channels=self.w4, out_channels=self.w4, step=encoder_downsamples[3])

        self.dconv3 = DoubleRandomDilatedConv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)
        self.dconv2 = DoubleRandomDilatedConv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)
        self.dconv1 = DoubleRandomDilatedConv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)
        self.dconv0 = DoubleRandomDilatedConv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

        self.upsample0 = nn.Upsample(scale_factor=upsamples_steps[0], mode='trilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=upsamples_steps[1], mode='trilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=upsamples_steps[2], mode='trilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=upsamples_steps[3], mode='trilinear', align_corners=True)

        self.dconv_last = nn.Conv3d(self.w1, self.final_in, (1, 1, 1), bias=True)
        self.threshold = ConfModel(self.w1, self.w1)
        # self.threshold = nn.Parameter(0.9*torch.ones(1))

    def forward(self, x, dilation_encoder=[1, 1, 1, 1], dilation_decoder=[1, 1, 1, 1]):
        # print(x.size())
        x0 = self.econv0(x)
        # print(x0.size())
        x1 = self.econv1(x0, dilation_encoder[0])
        # print(x1.size())
        x2 = self.econv2(x1, dilation_encoder[1])
        # print(x2.size())
        x3 = self.econv3(x2, dilation_encoder[2])
        # print(x3.size())
        x4 = self.bridge(x3, dilation_encoder[3])
        # print(x4.size())
        y = self.upsample0(x4)
        # print(y.size())
        # print(y.size())
        # print(x3.size())
        # if y.size()[3] != x3.size()[3]:
        #     diffY = torch.tensor([x3.size()[4] - y.size()[4]])
        #     diffX = torch.tensor([x3.size()[3] - y.size()[3]])
        #     y = F.pad(y, [0, diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        y3 = torch.cat([y, x3], dim=1)
        y3 = self.dconv3(y3, dilation_decoder[0])
        y2 = self.upsample1(y3)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.dconv2(y2, dilation_decoder[1])
        y1 = self.upsample2(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.dconv1(y1, dilation_decoder[2])
        y0 = self.upsample3(y1)
        y0 = torch.cat([y0, x0], dim=1)
        y0 = self.dconv0(y0, dilation_decoder[3])
        y = self.dconv_last(y0)
        pixel_thresholding = self.threshold(y0)

        return y, pixel_thresholding