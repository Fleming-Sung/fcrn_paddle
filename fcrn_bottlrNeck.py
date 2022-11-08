from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn

# from paddle.utils.download import get_weights_path_from_url

import math
import utils

class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UnPoolAsConv(nn.Layer):

    def __init__(self, in_channels, out_channels, batch_size):
        super(UnPoolAsConv, self).__init__()
        self.batch_size = batch_size

        # Convolution A (3x3)
        self.convA = nn.Conv2D(in_channels, out_channels, 3, padding='SAME')
        # Convolution B (2x3)
        self.convB = nn.Conv2D(in_channels, out_channels, (2, 3), padding='SAME')
        # Convolution C (3x2)
        self.convC = nn.Conv2D(in_channels, out_channels, (3, 2), padding='SAME')
        # Convolution D (2x2)
        self.convD = nn.Conv2D(in_channels, out_channels, 2, padding='SAME')

    def forward(self, input_data):
        # xA = input_data
        # outputA = self.convA(xA)
        # xB = nn.functional.pad(input_data, [1, 1, 0, 1])
        # outputB = self.convB(xB)
        # xC = nn.functional.pad(input_data, [0, 1, 1, 1])
        # outputC = self.convC(xC)
        # xD = nn.functional.pad(input_data, [0, 1, 0, 1])
        # outputD = self.convD(xD)

        outputA = self.convA(input_data)
        outputB = self.convB(input_data)
        outputC = self.convC(input_data)
        outputD = self.convD(input_data)

        # Interleaving elements of the four feature maps
        # --------------------------------------------------
        left = utils.interleave([outputA, outputB], axis=2)  # columns
        right = utils.interleave([outputC, outputD], axis=2)  # columns
        out = utils.interleave([left, right], axis=3)  # rows

        return out


class UpProject(nn.Layer):

    def __init__(self, in_channels, out_channels, batch_size):
        super(UpProject, self).__init__()
        self.batch_size = batch_size

        # branch 1
        self.unPool1 = UnPoolAsConv(in_channels, out_channels, batch_size)
        self.bn1_1 = nn.BatchNorm2D(out_channels)
        self.relu1_1 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels, out_channels, 3, padding='SAME')
        self.bn1_2 = nn.BatchNorm2D(out_channels)

        # branch 2
        self.unPool2 = UnPoolAsConv(in_channels, out_channels, batch_size)
        self.bn2_1 = nn.BatchNorm2D(out_channels)

        self.relu = nn.ReLU()

    def forward(self, input_data):
        out1 = self.unPool1(input_data)
        out1 = self.bn1_1(out1)
        out1 = self.relu1_1(out1)
        out1 = self.conv1(out1)
        out1 = self.bn1_2(out1)

        out2 = self.unPool2(input_data)
        out2 = self.bn2_1(out2)

        out = out1 + out2
        out = self.relu(out)

        return out


class FCRN(nn.Layer):

    def __init__(self, batch_size):
        super(FCRN, self).__init__()
        self.inplanes = 64
        self.batch_size = batch_size
        # b, 304, 228, 3
        # ResNet with out avrgpool & fc
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)  # b, 152 114, 64
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)  # b, 76, 57, 64
        self.layer1 = self._make_layer(Bottleneck, 64, 3)  # b, 76, 57, 256
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)  # b, 38, 29, 512
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)  # b, 19, 15, 1024
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)  # b, 10, 8, 2048

        # Up-Conv layers
        self.conv2 = nn.Conv2D(2048, 1024, kernel_size=1, bias_attr=False)# b, 10, 8, 1024
        self.bn2 = nn.BatchNorm2D(1024)

        self.up1 = self._make_upproj_layer(UpProject, 1024, 512, self.batch_size)
        self.up2 = self._make_upproj_layer(UpProject, 512, 256, self.batch_size)
        self.up3 = self._make_upproj_layer(UpProject, 256, 128, self.batch_size)
        self.up4 = self._make_upproj_layer(UpProject, 128, 64, self.batch_size)

        self.drop = nn.Dropout2D()

        self.conv3 = nn.Conv2D(64, 1, 3, padding='SAME')

        self.upsample = nn.Upsample((304, 228), mode='bilinear')

        # initialize
        initialize = False
        if initialize:
            for m in self.modules():
                if isinstance(m, nn.Conv2D):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2D):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_upproj_layer(self, block, in_channels, out_channels, batch_size):
        return block(in_channels, out_channels, batch_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.drop(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.upsample(x)

        return x

# from torchsummary import summary
# 测试网络模型


if __name__ == '__main__':
    batch_size = 1
    net = FCRN(batch_size)
    x = paddle.zeros(shape=[batch_size, 3, 304, 228])
    print(net(x))
    # paddle.summary(net, (1, 3, 304, 228))











