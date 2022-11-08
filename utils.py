import paddle
import numpy as np


class loss_huber(paddle.nn.Layer):
    def __init__(self):
        super(loss_huber, self).__init__()

    def forward(self, input, label):
        # b = input.shape[0]  # batch size
        c = input.shape[1]  # channel nums
        h = input.shape[2]  # height
        w = input.shape[3]  # width
        pred = paddle.reshape(input, (-1, c * h * w))
        truth = paddle.reshape(label, (-1, c * h * w))
        # 根据当前batch所有像素计算阈值
        t = 0.2 * paddle.max(paddle.abs(pred - truth))
        # 计算L1范数
        l1 = paddle.mean(paddle.mean(paddle.abs(pred - truth), 1), 0)
        # 计算论文中的L2
        l2 = paddle.mean(paddle.mean(((pred - truth)**2 + t**2) / t / 2, 1), 0)

        if l1 > t:
            return l2
        else:
            return l1


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
