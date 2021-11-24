import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# ==============================================================
# Different Components as layers in base network
# ==============================================================
class segnet_encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnet_encoder, self).__init__()
        self.convs_block = single_conv_bn(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.convs_block(inputs)
        unpooled_size = outputs.size()
        outputs, indices = self.maxpool(outputs)
        return outputs, indices, unpooled_size


class segnet_decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnet_decoder, self).__init__()
        self.convs_block = single_conv_bn(in_channels, out_channels)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.convs_block(outputs)
        return outputs


class segnet_last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnet_last, self).__init__()
        self.convs_block = last_conv(in_channels, out_channels)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.convs_block(outputs)
        return outputs


# ============================================================
def last_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
    )


def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )


def single_conv_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
        nn.ReLU6(inplace=True),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.ReLU6(inplace=True),
        nn.BatchNorm2d(out_channels, affine=True),
    )


def single_conv_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )


def depth_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 2, stride=2, padding=0),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


# ================================================================
# AMANet components
# ================================================================


# def mixed_attention_k1(in_channel, inter_channel):
#     return nn.Sequential(
#         nn.Conv2d(in_channel, inter_channel, 1, 1, 0),
#         nn.ReLU(),
#         nn.BatchNorm2d(inter_channel)
#     )

def mixed_attention_k3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, int(in_channel/8), 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(in_channel/8), out_channel, 1, 1, 0),
        nn.Sigmoid()
    )


def mixed_attention_gai(in_channel, inter_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel*4, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
        nn.InstanceNorm2d(out_channel*4, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel*4, out_channel*4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False, groups=out_channel*4),
        nn.InstanceNorm2d(out_channel * 4, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel*4, out_channel, 1, bias=False),
        nn.Sigmoid(),
    )

# ================================================================
# FA-FCN components
# ================================================================


def Self_Atten(in_channels, reso_scale_fac):
    return nn.Sequential(
        nn.Conv2d(in_channels, int(2*in_channels), reso_scale_fac, stride=reso_scale_fac, padding=0),
        nn.ReLU(inplace=True),
        nn.PixelShuffle(reso_scale_fac),
        nn.Conv2d(int(2*in_channels/(reso_scale_fac*reso_scale_fac)), in_channels, 1, stride=1, padding=0),
        nn.Sigmoid()
    )


def upsample_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, int(2*in_channels), kernel_size=1, stride=1, padding=0),
        nn.PixelShuffle(2),
        nn.Conv2d(int(2*in_channels/4), out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


def upsample_block_linear(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


def GlobalChannelAtten(input_reso, in_channels):
    return nn.Sequential(
        nn.AvgPool2d(input_reso),
        nn.Conv2d(in_channels, int(in_channels/8), 1, 1, 0),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(in_channels/8), in_channels, 1, 1, 0),
        nn.Sigmoid()
    )


# def Match_block(in_channels, out_channels):
#     return nn.Sequential(
#         nn.PixelShuffle(2),
#         nn.Conv2d(in_channels, out_channels, 1, 1, 0)
#     )


def output_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


def aunet_attention(in_channels):
    return nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, dilation=1, padding=1),
        nn.Sigmoid()
    )


def final_block(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, dilation=3, padding=3),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, 1, 1, 1, 0)
    )


# ========================================================
# MAU-net for the old IEEE TMI iUS submission
# ========================================================
def First_encoder(in_channels, width):
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(width, affine=True),
        nn.LeakyReLU(0.01),
        # nn.Conv2d(width, width, 3, stride=1, padding=1, bias=False)
    )


def Encoder(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.01),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.01),
    )


def Encoder_skip(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 2, stride=2, padding=0, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
    )


def Bridge(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.01),
        # nn.Dropout(0.5),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.01),
        # nn.Dropout(0.5),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(out_channels, in_channels, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(in_channels, affine=True),
        nn.LeakyReLU(0.01),
    )


def Decoder(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, int(in_channels/2), 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(int(in_channels/2), affine=True),
        nn.LeakyReLU(0.01),
        nn.Conv2d(int(in_channels / 2), int(in_channels / 2), 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(int(in_channels / 2), affine=True),
        nn.LeakyReLU(0.01),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(int(in_channels / 2), out_channels, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.01),
    )


def Last_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.01),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.01),
        nn.Conv2d(out_channels, 1, 1, stride=1, padding=0, bias=False)
    )


def MAU_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.01),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.Sigmoid()
    )


# =========================================================
def single_conv_in(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def single_conv_in_reluafter(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.InstanceNorm2d(out_channels, affine=True)
    )


def single_conv_in_bias(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )


def single_conv_in_reluafter_bias(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.InstanceNorm2d(out_channels, affine=True)
    )


def single_conv_bn_reluafter_bias(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels, affine=True)
    )


# ===============================================================================
# CBAM blocks
# Source: https://github.com/Youngkl0726/Convolutional-Block-Attention-Module/blob/master/CBAMNet.py
# Reference: https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
# ===============================================================================

class CBAM(nn.Module):

    def __init__(self, channels, reduction):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        new_module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = new_module_input * x + module_input
        return x


# ======================================================
# experiments about attention mechanisms that don't work
# ======================================================


'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class AnotherMixedAttn(nn.Module):
    def __init__(self, input_dim):
        super(AnotherMixedAttn, self).__init__()
        self.spatial_conv_query = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)

        self.channel_globalpool = nn.AdaptiveAvgPool2d(1)

        self.spatial_conv_key = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)
        self.spatial_key_bottleneck = nn.Conv2d(input_dim, 1, kernel_size=1, stride=1, dilation=1, padding=0, groups=1, bias=True)

        self.spatial_normalisation = nn.Softmax(2)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_smooth = nn.InstanceNorm2d(input_dim // 8)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()

        spatial_query = self.spatial_conv_query(x)
        channel_key = self.channel_globalpool(spatial_query)

        spatial_key = self.spatial_key_bottleneck(self.spatial_conv_key(x))
        key = torch.matmul(spatial_query.view(m_batchsize, channel * height, width), spatial_key.view(m_batchsize, height, width))
        #
        # spatial_key_norm = self.spatial_normalisation(key.view(m_batchsize, channel, width * height))
        attn_x = self.channel_expand(self.channel_highpass(self.channel_smooth(self.channel_squeeze(key.view(m_batchsize, channel, width, height) * channel_key))))

        attn_x = self.spatial_normalisation(attn_x.view(m_batchsize, channel, -1))

        attn_x = attn_x.view(m_batchsize, channel, width, height) * x

        return attn_x



'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class MixedNonLocal(nn.Module):
    def __init__(self, input_dim):
        super(MixedNonLocal, self).__init__()
        self.spatial_conv_query = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)

        self.channel_globalpool = nn.AdaptiveAvgPool2d(1)

        self.spatial_conv_key = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)
        self.spatial_key_bottleneck = nn.Conv2d(input_dim, 1, kernel_size=1, stride=1, dilation=1, padding=0, groups=1, bias=True)

        self.spatial_normalisation = nn.Softmax(2)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_smooth = nn.InstanceNorm2d(input_dim // 8)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()

        spatial_query = self.spatial_conv_query(x)
        channel_key = self.channel_globalpool(spatial_query)

        spatial_key = self.spatial_key_bottleneck(self.spatial_conv_key(x))
        key = torch.matmul(spatial_query.view(m_batchsize, channel*height, width), spatial_key.view(m_batchsize, height, width))


        spatial_key_norm = self.spatial_normalisation(key.view(m_batchsize, channel, width*height))

        attn_x = self.channel_expand(self.channel_highpass(self.channel_smooth(self.channel_squeeze(spatial_key_norm.view(m_batchsize, channel, width, height)*channel_key))))
        attn_x = attn_x * x
        return attn_x


'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class VerySparseNonLocalSqueezeExpand(nn.Module):
    def __init__(self, input_dim):
        super(VerySparseNonLocalSqueezeExpand, self).__init__()
        self.spatial_conv_query = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)

        self.channel_globalpool = nn.AdaptiveAvgPool2d(1)

        self.spatial_conv_key = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)
        self.spatial_key_bottleneck = nn.Conv2d(input_dim, 1, kernel_size=1, stride=1, dilation=1, padding=0, groups=1, bias=True)

        self.spatial_normalisation = nn.Softmax(1)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_smooth = nn.InstanceNorm2d(input_dim // 8)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()

        spatial_query = self.spatial_conv_query(x)
        channel_key = self.channel_globalpool(spatial_query)

        spatial_key = self.spatial_key_bottleneck(self.spatial_conv_key(x))
        spatial_key = torch.matmul(spatial_key.view(m_batchsize, height, width), spatial_key.view(m_batchsize, height, width).permute(0, 2, 1))
        spatial_key_norm = self.spatial_normalisation(spatial_key.view(m_batchsize, width*height, 1))

        reshaped_query = spatial_query.reshape(m_batchsize, channel*width, height)
        query_attn = torch.matmul(reshaped_query, spatial_key_norm.view(m_batchsize, width, height))

        attn_x = self.channel_expand(self.channel_highpass(self.channel_smooth(self.channel_squeeze(query_attn.view(m_batchsize, channel, width, height)*channel_key))))

        return attn_x


'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class FullSoftMaxPoolMixedAttn(nn.Module):
    def __init__(self, input_dim):
        super(FullSoftMaxPoolMixedAttn, self).__init__()
        self.spatial_conv_1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)
        self.spatial_conv_2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)

        self.normalisation_spatial_1 = nn.Softmax(1)
        self.normalisation_spatial_2 = nn.Softmax(1)

        self.globalpool = nn.AdaptiveAvgPool2d(1)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_instancenorm = nn.InstanceNorm2d(input_dim // 8, affine=True)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        # self.normalisation_final = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()
        spatial_key_1 = self.spatial_conv_1(x)
        spatial_key_2 = self.spatial_conv_2(x)
        channel_key = self.globalpool(spatial_key_1 + spatial_key_2)
        # This following loop needs to be optimised, too slow at the moment:
        for cc in range(channel):
            temp_key_1 = spatial_key_1[:, cc, :, :]
            temp_key_2 = spatial_key_2[:, cc, :, :]
            temp_attn_1 = self.normalisation_spatial_1(temp_key_1.view(m_batchsize, -1))
            temp_attn_2 = self.normalisation_spatial_2(temp_key_2.view(m_batchsize, -1))

            temp_attn = 0.5 * (temp_attn_1 + temp_attn_2)

            temp_attn = temp_attn.view(m_batchsize, 1, width, height)

            if cc == 0:
                spatial_attn = temp_attn
            else:
                spatial_attn = torch.cat([spatial_attn, temp_attn], dim=1)

        attn_x = self.channel_expand(self.channel_highpass(self.channel_instancenorm(self.channel_squeeze(spatial_attn * channel_key))))
        return attn_x


'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class LessSmoothSoftMaxPoolMixedAttn(nn.Module):
    def __init__(self, input_dim):
        super(LessSmoothSoftMaxPoolMixedAttn, self).__init__()
        self.spatial_conv_1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)
        self.spatial_conv_2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)

        self.normalisation_spatial_1 = nn.Softmax(1)
        self.normalisation_spatial_2 = nn.Softmax(1)

        self.globalpool = nn.AdaptiveAvgPool2d(1)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.normalisation_final = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()
        spatial_key_1 = self.spatial_conv_1(x)
        spatial_key_2 = self.spatial_conv_2(x)
        channel_key = self.globalpool(spatial_key_1 + spatial_key_2)
        # This following loop needs to be optimised, too slow at the moment:
        for cc in range(channel):
            temp_key_1 = spatial_key_1[:, cc, :, :]
            temp_key_2 = spatial_key_2[:, cc, :, :]
            temp_attn_1 = self.normalisation_spatial_1(temp_key_1.view(m_batchsize, -1))
            temp_attn_2 = self.normalisation_spatial_2(temp_key_2.view(m_batchsize, -1))
            temp_attn = 0.5*(temp_attn_1 + temp_attn_2)
            temp_attn = temp_attn.view(m_batchsize, 1, width, height)
            if cc == 0:
                spatial_attn = temp_attn
            else:
                spatial_attn = torch.cat([spatial_attn, temp_attn], dim=1)

        attn_x = self.normalisation_final(self.channel_expand(self.channel_highpass(self.channel_squeeze(spatial_attn * channel_key))))
        return attn_x


'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class HeavySoftMaxPoolMixedAttn(nn.Module):
    def __init__(self, input_dim):
        super(HeavySoftMaxPoolMixedAttn, self).__init__()
        self.spatial_conv_1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)
        self.spatial_conv_2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)
        self.channel_conv = nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=1, dilation=1, padding=2, groups=input_dim, bias=True)

        self.normalisation_spatial_1 = nn.Softmax(1)
        self.normalisation_spatial_2 = nn.Softmax(1)

        self.globalpool = nn.AdaptiveAvgPool2d(1)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_instancenorm = nn.InstanceNorm2d(input_dim // 8, affine=True)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.normalisation_final = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()
        spatial_key_1 = self.spatial_conv_1(x)
        spatial_key_2 = self.spatial_conv_2(x)
        channel_key = self.globalpool(self.channel_conv(x))
        # This following loop needs to be optimised, too slow at the moment:
        for cc in range(channel):
            temp_key_1 = spatial_key_1[:, cc, :, :]
            temp_key_2 = spatial_key_2[:, cc, :, :]
            temp_attn_1 = self.normalisation_spatial_1(temp_key_1.view(m_batchsize, -1))
            temp_attn_2 = self.normalisation_spatial_2(temp_key_2.view(m_batchsize, -1))
            temp_attn = 0.5*(temp_attn_1 + temp_attn_2)
            temp_attn = temp_attn.view(m_batchsize, 1, width, height)
            if cc == 0:
                spatial_attn = temp_attn
            else:
                spatial_attn = torch.cat([spatial_attn, temp_attn], dim=1)

        attn_x = self.normalisation_final(self.channel_expand(self.channel_highpass(self.channel_instancenorm(self.channel_squeeze(spatial_attn * channel_key)))))
        return attn_x


'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class LessSmoothHeavySoftMaxPoolMixedAttn(nn.Module):
    def __init__(self, input_dim):
        super(LessSmoothHeavySoftMaxPoolMixedAttn, self).__init__()
        self.spatial_conv_1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)
        self.spatial_conv_2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)
        self.channel_conv = nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=1, dilation=1, padding=2, groups=input_dim, bias=True)

        self.normalisation_spatial_1 = nn.Softmax(1)
        self.normalisation_spatial_2 = nn.Softmax(1)

        self.globalpool = nn.AdaptiveAvgPool2d(1)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.normalisation_final = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()
        spatial_key_1 = self.spatial_conv_1(x)
        spatial_key_2 = self.spatial_conv_2(x)
        channel_key = self.globalpool(self.channel_conv(x))
        # This following loop needs to be optimised, too slow at the moment:
        for cc in range(channel):
            temp_key_1 = spatial_key_1[:, cc, :, :]
            temp_key_2 = spatial_key_2[:, cc, :, :]
            temp_attn_1 = self.normalisation_spatial_1(temp_key_1.view(m_batchsize, -1))
            temp_attn_2 = self.normalisation_spatial_2(temp_key_2.view(m_batchsize, -1))
            temp_attn = 0.5*(temp_attn_1 + temp_attn_2)
            temp_attn = temp_attn.view(m_batchsize, 1, width, height)
            if cc == 0:
                spatial_attn = temp_attn
            else:
                spatial_attn = torch.cat([spatial_attn, temp_attn], dim=1)

        attn_x = self.normalisation_final(self.channel_expand(self.channel_highpass(self.channel_squeeze(spatial_attn * channel_key))))
        return attn_x



'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class NaiveSparseSeparateNonLocal(nn.Module):
    def __init__(self, input_dim):
        super(NaiveSparseSeparateNonLocal, self).__init__()
        self.spatial_conv_sparsekey = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)
        self.spatial_sparsekey_bottleneck = nn.Conv2d(input_dim, 1, kernel_size=1, stride=1, dilation=1, padding=0, groups=1, bias=True)
        self.spatial_normalisation = nn.Softmax(1)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_smooth = nn.InstanceNorm2d(input_dim // 8)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()
        spatial_key = self.spatial_sparsekey_bottleneck(self.spatial_conv_sparsekey(x))
        spatial_attn = self.spatial_normalisation(spatial_key.reshape(m_batchsize, width*height, 1))

        key = torch.matmul(x.reshape(m_batchsize, channel, width*height), spatial_attn)

        mixed_key = key.view(m_batchsize, channel, 1, 1)

        attn_x = self.channel_expand(self.channel_highpass(self.channel_smooth(self.channel_squeeze(mixed_key))))
        return attn_x


'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class MoreSparseSeparateNonLocal(nn.Module):
    def __init__(self, input_dim):
        super(MoreSparseSeparateNonLocal, self).__init__()
        self.spatial_conv_sparsequery = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)
        self.spatial_conv_sparsekey = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)
        self.spatial_sparsekey_bottleneck = nn.Conv2d(input_dim, 1, kernel_size=1, stride=1, dilation=1, padding=0, groups=1, bias=True)
        self.spatial_normalisation = nn.Softmax(1)

        self.channel_globalpool = nn.AdaptiveAvgPool2d(1)
        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_smooth = nn.InstanceNorm2d(input_dim // 8)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()
        spatial_key = self.spatial_sparsekey_bottleneck(self.spatial_conv_sparsekey(x))
        spatial_attn = self.spatial_normalisation(spatial_key.reshape(m_batchsize, width*height, 1))

        spatial_query = self.spatial_conv_sparsequery(x)

        key = torch.matmul(spatial_query.reshape(m_batchsize, channel, width*height), spatial_attn)

        mixed_key = self.channel_globalpool(spatial_query) * (key.view(m_batchsize, channel, 1, 1))

        attn_x = self.channel_expand(self.channel_highpass(self.channel_smooth(self.channel_squeeze(mixed_key))))
        return attn_x


'''
Spatial key: two sparse salience maps (kernel = 3, dilation = 3; kernel = 3, dilation = 6) at each channel
'''


class SeparateSparseMixedAttn(nn.Module):
    def __init__(self, input_dim):
        super(SeparateSparseMixedAttn, self).__init__()
        self.spatial_conv_sparsequery = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=3, padding=3, groups=input_dim, bias=True)
        self.spatial_conv_sparsekey = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, dilation=6, padding=6, groups=input_dim, bias=True)

        self.spatial_normalisation = nn.Softmax(1)

        self.channel_globalpool = nn.AdaptiveAvgPool2d(1)

        self.channel_squeeze = nn.Conv2d(input_dim, input_dim // 8, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.channel_smooth = nn.InstanceNorm2d(input_dim // 8)
        self.channel_highpass = nn.ReLU(inplace=True)
        self.channel_expand = nn.Conv2d(input_dim // 8, input_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

    def forward(self, x):
        m_batchsize, channel, width, height = x.size()

        spatial_sparse_query = self.spatial_conv_sparsequery(x)
        spatial_sparse_key = self.spatial_conv_sparsekey(x)

        # spatial_attn = self.spatial_normalisation(spatial_key.reshape(m_batchsize, width*height, 1))
        for cc in range(channel):
            temp_key_1 = spatial_sparse_query[:, cc, :, :]
            temp_key_2 = spatial_sparse_key[:, cc, :, :]
            temp_key_1 = temp_key_1.view(m_batchsize, width, height)
            temp_key_2 = temp_key_2.view(m_batchsize, width, height)
            temp_key = torch.matmul(temp_key_1, temp_key_2)
            temp_key = self.spatial_normalisation(temp_key.view(m_batchsize, -1))
            temp_key = temp_key.view(m_batchsize, 1, width, height)
            if cc == 0:
                spatial_attn = temp_key
            else:
                spatial_attn = torch.cat([spatial_attn, temp_key], dim=1)

        attn_x = self.channel_expand(self.channel_highpass(self.channel_smooth(self.channel_squeeze(spatial_attn * (self.channel_globalpool(spatial_sparse_key))))))
        return attn_x
