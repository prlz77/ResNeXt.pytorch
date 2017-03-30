from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality):
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * out_channels // 4
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0,
                                     groups=cardinality, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(in_channels, D, kernel_size=3, stride=stride, padding=0,
                                   groups=cardinality, bias=False)
        self.bn_conv = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                     groups=cardinality, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(x)
        bottleneck = F.relu(self.bn_conv.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(x)
        bottleneck = F.relu(self.bn_expand.forward(bottleneck), inplace=True)
        return F.relu(self.shortcut.forward(x) + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    def __init__(self, cardinality, depth, nlabels, widen_factor=4):
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.widen_factor = widen_factor
        self.nlabels = nlabels

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv_1_bn = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', 64, 1)
        self.stage_2 = self.block('stage_2', 128 * self.widen_factor, 2)
        self.stage_3 = self.block('stage_3', 256 * self.widen_factor, 2)
        self.classifier = nn.Linear(1024, nlabels)

    def block(self, name, in_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, in_channels * self.widen_factor,
                                                          pool_stride, self.cardinality))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(in_channels * self.widen_factor, in_channels * self.widen_factor,
                                                   1, self.cardinality))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.conv_1_bn.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = nn.AvgPool2d(8, 1)
        x = x.view(-1, 1024)
        return self.classifier(x)


if __name__ == '__main__':
    input = torch.zeros(2, 3, 40, 40)
    net = CifarResNeXt()