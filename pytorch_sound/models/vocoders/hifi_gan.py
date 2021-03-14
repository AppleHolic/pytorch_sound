#
# Reference : https://github.com/jik876/hifi-gan/blob/master/models.py
# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
# https://arxiv.org/abs/2010.05646
#
import torch
import torch.nn.functional as F
import torch.nn as nn
from argparse import Namespace
from pytorch_sound.models import register_model, register_model_architecture
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils import weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


@register_model('hifi_gan')
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i, upblock in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = upblock(x)
            xs = None
            for block in self.resblocks[i*self.num_kernels:(i+1)*self.num_kernels]:
                if xs is None:
                    xs = block(x)
                else:
                    xs += block(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


#
# I pick hifi-gan first
# 1. quality
#   - Hifi GAN V1 has a 4.3 MOS Score
#   - It seems also good quality on different samples.
# 2. Memory (about 50MB) and 2.5 times faster than real time on ryzen 3900.
# Thanks authors for sharing sources and checkpoints
#
@register_model_architecture('hifi_gan', 'hifi_gan_v1')
def hifi_gan_v1():
    return {
        'h': Namespace(
            **{
                'resblock': '1',
                'upsample_rates': [8, 8, 2, 2],
                'upsample_kernel_sizes': [16, 16, 4, 4],
                'upsample_initial_channel': 512,
                'resblock_kernel_sizes': [3, 7, 11],
                'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            }
        )
    }


#
# - Memory 3.6MB, super fast inference time.
# - It has similar size with multi-band melgan, but I think it has better quality.
#
@register_model_architecture('hifi_gan', 'hifi_gan_v2')
def hifi_gan_v2():
    return {
        'h': Namespace(
            **{
                'resblock': '1',
                'upsample_rates': [8, 8, 2, 2],
                'upsample_kernel_sizes': [16, 16, 4, 4],
                'upsample_initial_channel': 128,
                'resblock_kernel_sizes': [3, 7, 11],
                'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                'resblock_initial_channel': 64
            }
        )
    }


@register_model_architecture('hifi_gan', 'hifi_gan_v3')
def hifi_gan_v3():
    return {
        'h': Namespace(
            **{
                'resblock': '2',
                'upsample_rates': [8, 8, 4],
                'upsample_kernel_sizes': [16, 16, 8],
                'upsample_initial_channel': 256,
                'resblock_kernel_sizes': [3, 5, 7],
                'resblock_dilation_sizes': [[1, 2], [2, 6], [3, 12]]
            }
        )
    }
