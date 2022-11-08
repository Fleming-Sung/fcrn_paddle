import os
import paddle
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as transforms
# import scipy.io as scio


class NyuDepthLoader(Dataset):
    def __init__(self, data_path, lists):
        super(NyuDepthLoader, self).__init__()
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)
        # self.nyu = scio.loadmat(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(1, 2, 0)#.transpose(2, 1, 0)  # HWC
        dpt = self.dpts[img_idx]# .transpose(1, 0)
        img = Image.fromarray(img)
        dpt = Image.fromarray(dpt)
        input_transform = transforms.Compose([transforms.Resize(228),
                                              transforms.ToTensor()])

        target_depth_transform = transforms.Compose([transforms.Resize(228),
                                                     transforms.ToTensor()])

        img = input_transform(img)
        dpt = target_depth_transform(dpt)
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


# 数据集成载入
def load(subset = 'train'):
    batch_size = 16
    data_path = './data/nyu_depth_v2_labeled.mat'
    # 1.Load data
    train_lists, val_lists, test_lists = load_split()

    # print("Data setting...")

    if subset == 'train':
        lists = train_lists
    elif subset == 'validation':
        lists = val_lists
    elif subset == 'test':
        lists = test_lists

    dataloader = DataLoader(NyuDepthLoader(data_path, lists),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


# test
def test_loader():
    dataloader = load()
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

    # save
    # plot.imsave('input_rgb_epoch_0.png', input_rgb_image)
    # plot.imsave('gt_depth_epoch_0.png', input_gt_depth_image, cmap="viridis")


if __name__ == '__main__':
    test_loader()
















