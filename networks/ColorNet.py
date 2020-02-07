from torch import nn
from .EncodingBlocks import ResEncoder
from utils.AMDIMutils import flatten

class ColorNet(nn.Module):

    def __init__(self, n_rkhs, **kwargs):
        super(ColorNet, self).__init__()
        self.encoder = ResEncoder(n_rkhs=n_rkhs, **kwargs)
        self.layer_list = self.encoder.layer_list
        self.transpose_list = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_rkhs, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=4, stride=2, padding=1, dilation=1),
            nn.ConvTranspose2d(in_channels=384, out_channels=313, kernel_size=4, stride=2, padding=1, dilation=1),
        )

    def forward(self, x):
        layer_out = x
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_out
            layer_out = layer(layer_in)
        x = self.transpose_list(layer_out)
        return x

    def extract_feature(self, x):
        layer_out = x
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_out
            layer_out = layer(layer_in)
        return flatten(layer_out)


def create_model(**kwargs):
    return ColorNet(**kwargs)
