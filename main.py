'''
Re-implementation of paper Deeper Depth Prediction with Fully Convolutional Residual Networks
With paddle-paddle deep learning framework, single Nvidia GPU.
All items in one file, in order to make it suitable for ai_studio project.
@Fleming, Dec.5th.2022
'''


'''1. network modeling'''
import paddle
import math
import numpy as np
import paddle.nn as nn
from paddle.vision.models.resnet import resnet50
def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, paddle.Tensor):
        return incoming.shape
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")

def interleave(tensors, axis):
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return paddle.reshape(paddle.stack(tensors, axis + 1), new_shape)

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
        left = interleave([outputA, outputB], axis=2)  # columns
        right = interleave([outputC, outputD], axis=2)  # columns
        out = interleave([left, right], axis=3)  # rows

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

# net = FCRN(16)
# paddle.summary(net, (1, 3, 304, 228))


'''2. split and load data'''
# get split index
def load_split():
    # 读取事先设定的txt文件中的索引以进行数据集划分
    current_directoty = os.getcwd()
    train_lists_path = os.path.join(current_directoty, 'data', 'trainIdxs.txt')
    test_lists_path = os.path.join(current_directoty, 'data', 'testIdxs.txt')

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []

    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    # split 80% of train_data as train_data, while the rest 20% as validation data.
    val_start_idx = int(len(train_lists) * 0.8)
    val_lists = train_lists[val_start_idx:-1]
    train_lists = train_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists




# 3. configuration details of train process


# 4. training


# 5. evaluation


# 6. visualization
