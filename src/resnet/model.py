import torch.nn as nn
import torch.nn.functional as F


class ShortCut(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.pad = c_out - c_in
        self.stride = stride

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, 0, self.pad))


class ResBlock(nn.Module):
    '''
    out = F(x) + shortcut(x)
    '''

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3,
                      padding=1, stride=stride),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3,
                      stride=1, padding=1),
            nn.BatchNorm2d(c_out),
        )
        self.shortcut = ShortCut(c_in, c_out, stride)

    def forward(self, x):
        return F.relu(self.f(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, class_num=10):
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 16, 3,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        logits = self.resnet(x)
        return logits
