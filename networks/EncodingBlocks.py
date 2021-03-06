import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.AMDIMutils import flatten, Flatten


class MaybeBatchNorm2d(nn.Module):
    def __init__(self, n_ftr, affine, use_bn):
        super(MaybeBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(n_ftr, affine=affine)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return x


class NopNet(nn.Module):
    def __init__(self, norm_dim=None):
        super(NopNet, self).__init__()
        self.norm_dim = norm_dim

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = torch.sum(x ** 2., dim=self.norm_dim, keepdim=True)
            x_norms = torch.sqrt(x_norms + 1e-6)
            x = x / x_norms
        return x


class Conv3x3(nn.Module):
    def __init__(self, n_in, n_out, n_kern, n_stride, n_pad,
                 use_bn=True, pad_mode='constant'):
        super(Conv3x3, self).__init__()
        assert (pad_mode in ['constant', 'reflect'])
        self.n_pad = (n_pad, n_pad, n_pad, n_pad)
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(n_in, n_out, n_kern, n_stride, 0,
                              bias=(not use_bn))
        self.relu = nn.ReLU(inplace=True)
        self.bn = MaybeBatchNorm2d(n_out, True, use_bn) if use_bn else None

    def forward(self, x):
        if self.n_pad[0] > 0:
            # pad the input if required
            x = F.pad(x, self.n_pad, mode=self.pad_mode)
        # conv is always applied
        x = self.conv(x)
        # apply batchnorm if required
        if self.bn is not None:
            x = self.bn(x)
        # relu is always applied
        out = self.relu(x)
        return out


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class FakeRKHSConvNet(nn.Module):
    def __init__(self, n_input, n_output, use_bn=False):
        super(FakeRKHSConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=1, stride=1,
                               padding=0, bias=False)
        # BN is optional for hidden layer and always for output layer
        self.bn_hid = MaybeBatchNorm2d(n_output, True, use_bn)
        self.bn_out = MaybeBatchNorm2d(n_output, True, True)
        self.shortcut = nn.Conv2d(n_input, n_output, kernel_size=1,
                                  stride=1, padding=0, bias=True)
        # initialize shortcut to be like identity (if possible)
        if n_output >= n_input:
            eye_mask = np.zeros((n_output, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.shortcut.weight.data.uniform_(-0.01, 0.01)
            self.shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)
        return

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.)
        return

    def forward(self, x):
        h_res = self.conv2(self.relu1(self.bn_hid(self.conv1(x))))
        h = self.bn_out(h_res + self.shortcut(x))
        return h


class ConvResNxN(nn.Module):
    def __init__(self, n_in, n_out, width, stride, pad, use_bn=False):
        super(ConvResNxN, self).__init__()
        assert (n_out >= n_in)
        self.n_in = n_in
        self.n_out = n_out
        self.width = width
        self.stride = stride
        self.pad = pad
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_in, n_out, width, stride, pad, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, 1, 1, 0, bias=False)
        self.conv3 = None
        # ...
        self.bn1 = MaybeBatchNorm2d(n_out, True, use_bn)
        return

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.)
        return

    def forward(self, x):
        h1 = self.bn1(self.conv1(x))
        h2 = self.conv2(self.relu2(h1))
        if (self.n_out < self.n_in):
            h3 = self.conv3(x)
        elif (self.n_in == self.n_out):
            h3 = F.avg_pool2d(x, self.width, self.stride, self.pad)
        else:
            h3_pool = F.avg_pool2d(x, self.width, self.stride, self.pad)
            h3 = F.pad(h3_pool, (0, 0, 0, 0, 0, self.n_out - self.n_in))
        h23 = h2 + h3
        return h23


class ConvResBlock(nn.Module):
    def __init__(self, n_in, n_out, width, stride, pad, depth, use_bn):
        super(ConvResBlock, self).__init__()
        layer_list = [ConvResNxN(n_in, n_out, width, stride, pad, use_bn)]
        for i in range(depth - 1):
            layer_list.append(ConvResNxN(n_out, n_out, 1, 1, 0, use_bn))
        self.layer_list = nn.Sequential(*layer_list)
        return

    def init_weights(self, init_scale=1.):
        '''
        Do a fixup-style init for each ConvResNxN in this block.
        '''
        for m in self.layer_list:
            m.init_weights(init_scale)
        return

    def forward(self, x):
        # run forward pass through the list of ConvResNxN layers
        x_out = self.layer_list(x)
        return x_out


class ResEncoder(nn.Module):
    def __init__(self, num_channels=3, ndf=64, n_rkhs=512,
                 n_depth=3, encoder_size=32, use_bn=False):
        super(ResEncoder, self).__init__()
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None

        # encoding block for local features
        print('Using a {}x{} encoder'.format(encoder_size, encoder_size))
        if encoder_size == 32:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 4, True, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 64:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 128:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 5, 2, 2, False, pad_mode='reflect'),
                Conv3x3(ndf, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(encoder_size))

    def forward(self, x):
        layer_out = x
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_out
            layer_out = layer(layer_in)
        return flatten(layer_out)
