from torchvision.models import resnet18
from torch import nn
import torch.nn.functional as F


class Resnet18(nn.Module):

    def __init__(self, num_classes, num_stages, num_hidden, **kwargs):
        super(Resnet18, self).__init__()
        self.res = nn.Sequential(*list(resnet18(**kwargs).children())[:-1])
        self.hidden = nn.Sequential(nn.Linear(512, num_hidden), nn.ReLU())
        self.classifier = nn.Linear(num_hidden, num_classes)

    def forward(self, input):
        x = self.extract_feature(input)
        x = self.classifier(x)
        return F.softmax(x, dim=1)

    def extract_feature(self, input):
        b = input.size(0)
        x = self.res(input)
        x = x.view(b, -1)
        x = self.hidden(x)
        return x


def create_model(**kwargs):
    # return resnet18()
    return Resnet18(**kwargs)
