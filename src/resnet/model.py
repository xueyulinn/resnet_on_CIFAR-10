import torch.nn as nn
import torch.nn.functional as F


class ShortCut(nn.Module):

    '''
    Option A
    '''

    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.pad = c_out - c_in
        self.stride = stride

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, 0, self.pad))


class Block(nn.Module):
    '''
    Each block has 2 3x3 convs
    block out = F(x) + shortcut(x)
    '''

    def __init__(self, c_in, c_out, stride=1, residual=True):
        super().__init__()
        self.residual = residual
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
        return F.relu(self.f(x) + self.shortcut(x)) if self.residual else F.relu(self.f(x))


class ResNet(nn.Module):
    def __init__(self, n, class_num=10, residual=True):
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 16, 3,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 6n in total
            self.make_stage(16, 16, n, residual=residual),
            self.make_stage(16, 32, n, 2, residual=residual),
            self.make_stage(32, 64, n, 2, residual=residual),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, class_num)
        )

    def make_stage(self, c_in, c_out, n, stride=1, residual=True):
        '''
        Each stage has n blocks hence 2*n layers 
        '''
        blocks = [Block(c_in, c_out, stride, residual)]
        blocks += [Block(c_out, c_out, 1, residual) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.resnet(x)
