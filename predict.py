import argparse
import os
import paddle
import numpy as np
from paddle.io import Dataset
from matplotlib import pyplot as plt
from PIL import Image
import fcrn


def predict(model_data_path, image_path):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    class TestImage(Dataset):
        def __init__(self, num_samples):
            super(TestImage, self).__init__()
            self.num_samples = num_samples
            # self.return_label = return_label
            # 在 `__init__` 中定义数据增强方法，此处为调整图像大小
            # self.transform = Compose([Resize(size=32)])

        def __getitem__(self, index):
            # data = paddle.uniform(IMAGE_SIZE, dtype='float32')
            # 在 `__getitem__` 中对数据集使用数据增强方法
            # data = self.transform(data.numpy())
            # Read image
            img1 = Image.open(image_path)
            img2 = img1.resize([width, height], Image.ANTIALIAS)
            img3 = np.array(img2).astype('float32')
            if img3.shape[-1] == 4:
                img4 = np.delete(img3, 3, axis=-1)
            elif img3.shape[-1] == 3:
                img4 = img3
            else:
                return 1
            data = img4
            trans = paddle.vision.transforms.Transpose()
            data = trans(data)
            # label = paddle.randint(0, CLASS_NUM - 1, dtype='int64')

            return data

        def __len__(self):
            return self.num_samples

    test_dataset = TestImage(1)

    net = fcrn.FCRN(batch_size)
    model = paddle.Model(net)
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # model.load(model_data_path)
    # './models/NYU_ResNet-UpProj.npy'
    model.prepare(
        optim,
        paddle.nn.loss.CrossEntropyLoss()
    )

    pred = model.predict(test_dataset)

    # Plot result
    fig = plt.figure()
    ii = plt.imshow(np.array(pred[0][0])[0, 0, :, :], interpolation='nearest')
    plt.imsave('testResult1.jpg', np.array(pred[0][0])[0, 0, :, :])
    fig.colorbar(ii)
    plt.show()

    return pred


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    # print(pred)
    # tu_pred = pred[0]
    # print(tu_pred)
    # print(type(tu_pred), len(tu_pred))
    # ar_pred = np.array(tu_pred[0])
    # print(ar_pred)
    # print(np.shape(ar_pred))
    # # ppred = ar_pred[0, :, :, 0]
    # # print(ppred)


    os._exit(0)


if __name__ == '__main__':
    main()
