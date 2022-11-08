from __future__ import division
from __future__ import print_function
import math
import utils
import paddle
import paddle.nn as nn
from paddle.vision.models.resnet import resnet50
# from paddle.utils.download import get_weights_path_from_url
from paddle.vision.models.resnet import BottleneckBlock
__all__ = []


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
        # ImageNet dataset Pretrianed ResNet without avrgpool & fc
        self.resNet50 = resnet50(pretrained=True, num_classes=0, with_pool=False)

        # Up-Conv layers
        self.conv1 = nn.Conv2D(2048, 1024, kernel_size=1, bias_attr=False)  # b, 10, 8, 1024
        self.bn1 = nn.BatchNorm2D(1024)

        self.up1 = self._make_upproj_layer(UpProject, 1024, 512, self.batch_size)
        self.up2 = self._make_upproj_layer(UpProject, 512, 256, self.batch_size)
        self.up3 = self._make_upproj_layer(UpProject, 256, 128, self.batch_size)
        self.up4 = self._make_upproj_layer(UpProject, 128, 64, self.batch_size)

        self.drop = nn.Dropout2D()

        self.conv2 = nn.Conv2D(64, 1, 3, padding='SAME')

        self.relu = nn.ReLU()

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

    def _make_upproj_layer(self, block, in_channels, out_channels, batch_size):
        return block(in_channels, out_channels, batch_size)

    def forward(self, x):
        x = self.resNet50(x)

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.drop(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.upsample(x)

        return x

# from torchsummary import summary
# 测试网络模型


if __name__ == '__main__':
    batch_size = 1
    net = FCRN(batch_size)
    x = paddle.zeros(shape=[batch_size, 3, 304, 228])
    # print(net(x))
    paddle.summary(net, (1, 3, 304, 228))











