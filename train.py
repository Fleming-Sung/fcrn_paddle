import fcrn
import loader
import myLoader
import pngLoader
import utils
import paddle
import zipfile
from pathlib import Path


# 0.Initial Index
batch_size = 16
num_epochs = 10
start_epoch = 0
my_lr = 1.0e-4
ori_data_dir = Path('./data/nyu_v2/nyu_v2.zip')
# unzipped_dir = Path('./sy')
my_data_dir = Path('./nyuv2/extracted')

# 1.load data
print('Loading data...')

# if not my_data_dir.exists():
#     print('Unzipping dataset...')
#     with zipfile.ZipFile(ori_data_dir) as zf:
#         zf.extractall()

# train_data = loader.load(subset='train')
# test_data = loader.load(subset='validation')

train_data = pngLoader.load(my_data_dir, batch_size, subset='train')
vali_data = pngLoader.load(my_data_dir, batch_size, subset='validation')

# 2. load model
print('Loading model...')
net = fcrn.FCRN(batch_size)
model = paddle.Model(net)

# 3. Loss
print('Loss_fn settint...')
huber_loss = utils.loss_huber()

# 4. Optim
print('Optimizer setting...')
adam = paddle.optimizer.Adam(learning_rate=my_lr, parameters=model.parameters())

# 5. Train
# 为模型训练做准备，设置优化器，损失函数和精度计算方式
model.prepare(optimizer=adam,
              loss=huber_loss,
              metrics=paddle.metric.Accuracy())

# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
model.fit(train_data,
          epochs=num_epochs,
          batch_size=batch_size,
          verbose=1)

# 用 evaluate 在测试集上对模型进行验证
eval_result = model.evaluate(vali_data, verbose=1)


