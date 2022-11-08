import paddle
import os
from paddle.io import Dataset, DataLoader
from PIL import Image
import numpy as np
import paddle.vision.transforms as transforms
import matplotlib.pyplot as plt


class PngLoader(Dataset):
    def __init__(self, png_path, lists):
        super(PngLoader, self).__init__()
        self.png_path = png_path
        self.lists = lists

    def __getitem__(self, index):
        img_idx = self.lists[index]
        dpt_dir = os.path.join(self.png_path, 'nyu_depths')
        img_dir = os.path.join(self.png_path, 'nyu_images')
        img_name = str(img_idx)+'.jpg'
        dpt_name = str(img_idx) + '.png'
        img_path = os.path.join(img_dir, img_name)
        dtp_path = os.path.join(dpt_dir, dpt_name)

        img_ori = Image.open(img_path)
        dpt_ori = Image.open(dtp_path)

        input_transform = transforms.Compose([transforms.Resize(228),
                                              transforms.ToTensor()])

        target_depth_transform = transforms.Compose([transforms.Resize(228),
                                                     transforms.ToTensor()])
        img = input_transform(img_ori)
        dpt = target_depth_transform(dpt_ori)

        img = paddle.transpose(img, perm=[0, 2, 1])
        dpt = paddle.transpose(dpt, perm=[0, 2, 1])
        return img, dpt

    def __len__(self):
        return len(self.lists)


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


# pngpath = './nyu_v2'
# 数据集成载入
def load(png_path, batch_size, subset = 'train'):
    # batch_size = 16

    # 1.Load data
    train_lists, val_lists, test_lists = load_split()

    # print("Data setting...")

    if subset == 'train':
        lists = train_lists
    elif subset == 'validation':
        lists = val_lists
    elif subset == 'test':
        lists = test_lists

    dataloader = DataLoader(PngLoader(png_path, lists),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


# test
def test_loader():
    png_path = './nyu_v2'
    batch_size = 16
    dataloader = load(png_path, batch_size)
    for i, (image, depth) in enumerate(dataloader):
        print('The original shape of a batch is:')
        print(image.shape)
        print(depth.shape)
        print('\n')
        if i == 0:
            break

    input_rgb_image = image[0]  # .transpose(1, 2, 0)
    input_rgb_image_trans = paddle.transpose(input_rgb_image, perm=[1, 2, 0])
    input_gt_depth_image = depth[0][0]  # .numpy().astype(np.float32)

    print('The original shape of a picture is:')
    print(input_rgb_image.shape, '\n')
    print('The transposed shape of a picture is:')
    print(input_rgb_image_trans.shape)
    print(input_gt_depth_image.shape)
    plt.imshow(input_rgb_image_trans)
    plt.show()
    plt.imshow(input_gt_depth_image, cmap="viridis")
    plt.show()


if __name__ == '__main__':
    test_loader()

