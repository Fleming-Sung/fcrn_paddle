import os

import paddle
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock
import paddle.nn as nn

# import utils
#
# a = paddle.zeros([1, 512, 18, 6])
# b = paddle.zeros([1, 512, 18, 6])
#
# b = utils.interleave([a, b], axis=2)
# print(b.shape)

# x = paddle.zeros([2, 4, 5, 5])
# conv1 = nn.Conv2D(4, 6, 2, 1)
# y = conv1(x)
#
# print(y)
# print('shape of y = ', y.shape)
# print('shape of y_dim3 = ', y.shape[2])
# net = ResNet(BottleneckBlock, 50, num_classes=0, with_pool=False)
# x = paddle.zeros(shape=[1, 3, 304, 228])
# print(net(x))

# resnet18 = ResNet(BasicBlock, 18)

# class upProj():
#
#     def __init__(self):
#         super(upProj, self).__init__()
#
#
#
#
# class fcrn(paddle.nn.Layer):
#
#     def __init__(self):
#         super(fcrn, self).__init__()
#
#         self.resnet = ResNet(BottleneckBlock, 50, num_classes=0, with_pool=False)
#         self.upProj = upProj()

# resnet不包括最后的全连接层和池化层时的调用
# fcrn_res = ResNet(BottleneckBlock, 50, num_classes=0, with_pool=False)
# paddle.summary(fcrn_res, (1, 3, 304, 228))
#
# x = paddle.rand([1, 3, 224, 224])
# out = resnet18(x)

# print(out.shape)

# from PIL import Image
# import numpy as np
#
#
# # Default input size
# height = 228
# width = 304
# channels = 3
# batch_size = 1
# img1 = Image.open('assets/test.jpg')
# img2 = img1.resize([width, height], Image.ANTIALIAS)
# img3 = np.array(img2).astype('float32')
# if img3.shape[-1] == 4:
#     img3 = np.delete(img3, 3, axis=-1)
#
# print(img3.shape)
# import numpy as np
# from PIL import Image
# from paddle.vision.transforms import Resize
#
# transform = Resize(size=224)
#
# fake_img = Image.fromarray((np.random.rand(100, 120, 3) * 255.).astype(np.uint8))
# print(fake_img)
# trans_img = transform(fake_img)
# print(trans_img.size)
# import zipfile
#
# with zipfile.ZipFile('./data/nyu_v2/nyu_v2.zip') as zf:
#     zf.extractall()
#
# path = './'
# for item in os.listdir(path):
#     if os.path.isdir(os.path.join(path, item)):
#         i = 1
#         for i:
from matplotlib import pyplot as plt
from PIL import Image
import paddle.vision.transforms as transforms

dpt_test = Image.open('./nyuv2/extracted/dpt_00000.png')
print(dpt_test)
tr = transforms.ToTensor()
dpt_test_tr = tr(dpt_test)
print(dpt_test_tr)
print(dpt_test_tr.shape)
for ts in dpt_test_tr:
    print(ts)
    for ls in ts:
        print(ls)
        # for em in ls:
        #     print(em)
