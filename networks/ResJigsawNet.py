from torchvision.models import resnet18
from torch import nn
import torch as t
import torch.nn.functional as F


class ResJigsaw(nn.Module):

    def __init__(self, perm, num_classes, num_hidden, **kwargs):
        super(ResJigsaw, self).__init__()

        self.alex = nn.Sequential(*list(resnet18(**kwargs).children())[:-1])
        self.perm = perm
        self.parts = perm * perm
        self.fc7 = nn.Sequential(nn.Linear(self.parts * 512, num_hidden), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(num_hidden, num_classes))

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.classifier(x)
        return x

    def extract_feature(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []
        for i in range(self.parts):
            z = self.alex(x[i])
            z = z.view([B, 1, -1])
            x_list.append(z)
        x = t.cat(x_list, 1)
        x = self.fc7(x.view(B, -1))
        return x


def create_model(**kwargs):
    # return resnet18()
    return ResJigsaw(**kwargs)
