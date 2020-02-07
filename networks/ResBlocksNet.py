from .EncodingBlocks import ResEncoder
from torch import nn


class ResBlocksNet(nn.Module):

    def __init__(self, num_classes, n_rkhs, **kwargs):
        super().__init__()
        self.encoder = ResEncoder(n_rkhs=n_rkhs,**kwargs)
        self.classifier = nn.Linear(n_rkhs, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

    def extract_feature(self, x):
        return self.encoder(x)


def create_model(**kwargs):
    # return resnet18()
    return ResBlocksNet(**kwargs)
