import fcrn
# import loader
# import myLoader
import pngLoader
import utils
import paddle
import zipfile
from pathlib import Path


# 0.Initial Index
batch_size = 16
num_epochs = 5000
start_epoch = 0
my_lr = 0.0001
monentum = 0.9
weight_decay = 0.001
# ori_data_dir = Path('./data/nyu_depth_v2_labeled.mat')
# unzipped_dir = Path('./sy')
my_data_dir = Path('./data/nyu_depth_v2_labeled.mat')

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

# # load pre-trained model
# model.load_state_dict(load_weights(model, weights_file, dtype)) #加载官方参数，从tensorflow转过来
#     #加载训练模型
#     resume_from_file = False
#     resume_file = './model/model_300.pth'
#     if resume_from_file:
#         if os.path.isfile(resume_file):
#             checkpoint = torch.load(resume_file)
#             start_epoch = checkpoint['epoch']
#             model.load_state_dict(checkpoint['state_dict'])
#             print("loaded checkpoint '{}' (epoch {})"
#                   .format(resume_file, checkpoint['epoch']))
#         else:
#             print("can not find!")

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
          verbose=1,
          save_freq=300,
          save_dir='./model')

# 用 evaluate 在测试集上对模型进行验证
eval_result = model.evaluate(vali_data, verbose=1)


