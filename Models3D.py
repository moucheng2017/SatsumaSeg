import torch
# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F


class ThresholdEncoder(nn.Module):
    def __init__(self, c=8, ratio=8):
        '''
        Args:
            c:
            ratio:
        We assume the latent variable confidence threshold is gaussian distribution
        '''
        super(ThresholdEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Conv3d(in_channels=c, out_channels=c*ratio, kernel_size=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.InstanceNorm3d(c*ratio, affine=True),
            nn.PReLU(),
            nn.Conv3d(in_channels=c*ratio, out_channels=c * ratio, kernel_size=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.InstanceNorm3d(c * ratio, affine=True),
            nn.PReLU()
        )

        self.threshold_logvar = nn.Conv3d(in_channels=ratio*c, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.threshold_mean = nn.Conv3d(in_channels=ratio*c, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 0, 0), bias=True)

        # self.softplus = nn.Softplus()

    def forward(self, x):
        y = self.network(x)
        # return self.softplus(self.threshold_mean(y)), self.threshold_logvar(y)
        return self.threshold_mean(y), self.threshold_logvar(y)


# class ThresholdDecoder(nn.Module):
#     def __init__(self, c=8, ratio=8):
#         '''
#         Args:
#             c:
#             ratio:
#         '''
#         super(ThresholdDecoder, self).__init__()
#
#         self.network = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=c*ratio, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(1, 1), bias=False),
#             nn.InstanceNorm2d(c*ratio, affine=True),
#             nn.PReLU(),
#             nn.Conv2d(in_channels=c*ratio, out_channels=c * ratio, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(1, 1), bias=False),
#             nn.InstanceNorm2d(c * ratio, affine=True),
#             nn.PReLU()
#         )
#
#         self.threshold_logvar = nn.Conv2d(in_channels=ratio*c, out_channels=1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=True)
#         self.threshold_mean = nn.Conv2d(in_channels=ratio*c, out_channels=1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0), bias=True)
#
#     def forward(self, x):
#         y = self.network(x)
#         y = torch.mean(y, dim=-1, keepdim=True)
#         y = torch.mean(y, dim=-2, keepdim=True)
#         return self.threshold_mean(y), self.threshold_logvar(y)


# class UnetBPL3D(nn.Module):
#     def __init__(self,
#                  in_ch,
#                  width,
#                  depth,
#                  out_ch,
#                  norm='in',
#                  ratio=8,
#                  # detach=True
#                  ):
#         '''
#         Args:
#             in_ch:
#             width:
#             depth:
#             out_ch:
#             norm:
#             ratio:
#             detach:
#         '''
#         super(UnetBPL, self).__init__()
#         if out_ch == 2:
#             out_ch = 1
#         # self.detach_bpl = detach
#         self.segmentor = Unet(in_ch, width, depth, out_ch, norm=norm, side_output=True)
#         self.encoder = ThresholdEncoder(c=width, ratio=ratio)
#         self.avg_pool = nn.AvgPool2d(1)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
#
#     def forward(self, x):
#         outputs_dict = self.segmentor(x)
#         output, threshold_input = outputs_dict.get('segmentation'), outputs_dict.get('side_output')
#
#         # if self.detach_bpl == 1:
#         #     mu, logvar = self.encoder(threshold_input.detach())
#         # else:
#         #     mu, logvar = self.encoder(threshold_input)
#
#         mu, logvar = self.encoder(self.avg_pool(threshold_input))
#
#         learnt_threshold = self.reparameterize(mu, logvar)
#         # learnt_threshold = F.softplus(learnt_threshold)
#         # learnt_threshold = torch.sigmoid(learnt_threshold)
#
#         return {
#             'segmentation': output,
#             'mu': mu,
#             'logvar': logvar,
#             'learnt_threshold': learnt_threshold
#                 }


# class Unet3D(nn.Module):
#     def __init__(self,
#                  in_ch,
#                  width,
#                  depth,
#                  classes,
#                  norm='in',
#                  side_output=False):
#         '''
#         Args:
#             in_ch:
#             width:
#             depth:
#             classes:
#             norm:
#             side_output:
#         '''
#         super(Unet3D, self).__init__()
#
#         assert depth > 1
#         self.depth = depth
#
#         # if classes == 2:
#         #     classes = 1
#
#         self.side_output_mode = side_output
#
#         self.decoders = nn.ModuleList()
#         self.encoders = nn.ModuleList()
#
#         for i in range(self.depth):
#
#             if i == 0:
#                 self.encoders.append(double_conv_3d(in_channels=in_ch, out_channels=width, step=1, norm=norm))
#                 self.decoders.append(double_conv_3d(in_channels=width*2, out_channels=width, step=1, norm=norm))
#             elif i < (self.depth - 1):
#                 self.encoders.append(double_conv_3d(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
#                 self.decoders.append(double_conv_3d(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))
#
#             else:
#                 self.encoders.append(double_conv_3d(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
#                 self.decoders.append(double_conv_3d(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#         self.conv_last = nn.Conv3d(width, classes, 1, bias=True)
#
#     def forward(self, x):
#
#         y = x
#         encoder_features = []
#
#         for i in range(len(self.encoders)):
#
#             y = self.encoders[i](y)
#             encoder_features.append(y)
#
#         for i in range(len(encoder_features)):
#
#             y = self.upsample(y)
#             y_e = encoder_features[-(i+1)]
#
#             # if y_e.shape[-1] != y.shape[-1]:
#             #     diffY = torch.tensor([y_e.size()[-1] - y.size()[-1]])
#             #     diffX = torch.tensor([y_e.size()[-2] - y.size()[-2]])
#             #     # y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
#             #     y = F.pad(y, [torch.div(diffX, 2, rounding_mode='floor'), diffX - torch.div(diffX, 2, rounding_mode='floor'), torch.div(diffY, 2, rounding_mode='floor'), diffY - torch.div(diffY, 2, rounding_mode='floor')])
#
#             y = torch.cat([y_e, y], dim=1)
#             y = self.decoders[-(i+1)](y)
#
#         output = self.conv_last(y)
#
#         if self.side_output_mode is False:
#             return {'segmentation': output}
#         else:
#             return {'segmentation': output,
#                     'side_output': y}

def double_conv(in_channels, out_channels, step):
    # double convolutional layers
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, (3, 3, 3), stride=step, padding=(1, 1, 1), groups=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1 ,1), groups=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def single_conv(in_channels, out_channels, step):
    # single convolutional layers
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, (3, 3, 3), stride=step, padding=(1, 1, 1), groups=1, bias=False),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


class DoubleRandomDilatedConv(nn.Module):
    # Random dilation convolutional layers
    def __init__(self, in_channels, out_channels, step):
        super(DoubleRandomDilatedConv, self).__init__()
        self.attention_branch = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=step, dilation=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, random_seed):
        self.attention_branch[0].dilation = (1, int(random_seed), int(random_seed))
        self.attention_branch[0].padding = (1, int(random_seed), int(random_seed))
        self.attention_branch[3].dilation = (1, int(random_seed), int(random_seed))
        self.attention_branch[3].padding = (1, int(random_seed), int(random_seed))
        output = self.attention_branch(x)
        return output


class Unet3D(nn.Module):
    # Stochastic Receptive Field Net
    def __init__(self, in_ch, width, classes, depth=4, z_downsample=4, norm='in', side_output=False):
        super(Unet3D, self).__init__()

        self.final_in = classes

        self.w1 = width
        self.w2 = width * 2
        self.w3 = width * 4
        self.w4 = width * 8

        self.econv0 = single_conv(in_channels=in_ch, out_channels=self.w1, step=1)
        self.econv1 = DoubleRandomDilatedConv(in_channels=self.w1, out_channels=self.w2, step=(2, 2, 2))
        self.econv2 = DoubleRandomDilatedConv(in_channels=self.w2, out_channels=self.w3, step=(2, 2, 2))
        self.econv3 = DoubleRandomDilatedConv(in_channels=self.w3, out_channels=self.w4, step=(2, 2, 2))
        self.bridge = DoubleRandomDilatedConv(in_channels=self.w4, out_channels=self.w4, step=(2, 2, 2))

        self.dconv3 = DoubleRandomDilatedConv(in_channels=self.w4+self.w4, out_channels=self.w3, step=1)
        self.dconv2 = DoubleRandomDilatedConv(in_channels=self.w3+self.w3, out_channels=self.w2, step=1)
        self.dconv1 = DoubleRandomDilatedConv(in_channels=self.w2+self.w2, out_channels=self.w1, step=1)
        self.dconv0 = DoubleRandomDilatedConv(in_channels=self.w1+self.w1, out_channels=self.w1, step=1)

        self.upsample0 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)

        self.dconv_last = nn.Conv3d(self.w1, self.final_in, (1, 1, 1), bias=True)

    def forward(self, x, dilation_encoder=(1, 1, 1, 1), dilation_decoder=(1, 1, 1, 1)):
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
        return {'segmentation': y}


def double_conv_3d(in_channels, out_channels, step, norm):
    '''
    Args:
        in_channels:
        out_channels:
        step:
        norm:
    Returns:
    '''
    if norm == 'in':
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'bn':
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.BatchNorm3d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm3d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'ln':
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'gn':
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU()
        )