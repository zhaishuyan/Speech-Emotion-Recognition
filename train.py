import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from keras.utils import np_utils

import models
from models.dnn.dnn import EmotionDataset
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import utils.opts as opts

import torch
import torch.nn as nn
import torch.utils.data as Data

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def save_checkpoint(state, epoch, loss, ext='pth.tar'):
    check_path = os.path.join(config.checkpoint_path,
                              f'ctpn_ep{epoch:02d}_'
                              f'{loss:.4f}.{ext}')
    torch.save(state, check_path)
    print('saving to {}'.format(check_path))


def train_model(config, x_train, y_train, x_val=None, y_val=None,
                batch_size=32, n_epochs=50):
    '''

    '''

    '''
    定义model, optimizer, loss function
    '''
    model = models.setup(config=config, n_feats=x_train.shape[1])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    device = torch.device('cuda:0')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_func = nn.CrossEntropyLoss()
    # self.model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])

    '''
    加载数据
    '''

    if x_val is None or y_val is None:
        x_test, y_val = x_train, y_train

    x_train, x_val = np.expand_dims(x_train, 1), np.expand_dims(x_val, 1)

#     print(x_train.shape)
#     print(batch_size)

    train_data = EmotionDataset(x_train, y_train)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # 采用两个进程来提取
    )

    test_data = EmotionDataset(x_val, y_val)
    test_x = torch.tensor(test_data.x, dtype=torch.float32).cuda()
    test_y = torch.tensor(test_data.y)
    test_y = torch.max(test_y, -1)[1].data.numpy().reshape(1, -1)[0]

    '''
    train model
    '''
    for epoch in range(config.epochs):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = torch.tensor(batch_x, dtype=torch.float32).cuda()
            # 训练
            model.train()
            batch_y_pred = model(batch_x)
            batch_y_pred = batch_y_pred.squeeze()
            # batch_y_pred = torch.max(batch_y_pred,-1)[1].reshape(1,-1)[0]
            batch_y = torch.max(batch_y, -1)[1]
            batch_y = torch.tensor(batch_y, dtype=torch.long).cuda()
            loss = loss_func(batch_y_pred, batch_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        # 验证
        model.eval()
        with torch.no_grad():
            test_output = model(test_x)
            pred_y = torch.max(test_output, -1)[1].cpu().data.numpy().reshape(1, -1)[0]
            sum = 0.
            for i in range(test_y.shape[0]):
                if pred_y[i] == test_y.data[i]:
                    sum += 1.

            # #记录一种计算accuracy的方法
            # red = logits.argmax(dim=1)
            # num_correct += torch.eq(pred, y).sum().float().item()

            accuracy = sum / test_y.shape[0]


        # 输出
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)
    save_checkpoint(state={'model_state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict(),
                           'epoch': epoch},
                    epoch=epoch,
                    loss=loss)


'''
train(): 训练模型

输入:
	config(Class)

输出:
    model: 训练好的模型
'''


def train(config):
    # 加载被 preprocess.py 预处理好的特征
    if (config.feature_method == 'o'):
        x_train, x_test, y_train, y_test = of.load_feature(config, config.train_feature_path_opensmile, train=True)

    elif (config.feature_method == 'l'):
        x_train, x_test, y_train, y_test = lf.load_feature(config, config.train_feature_path_librosa, train=True)

    # x_train, x_test (n_samples, n_feats)
    # y_train, y_test (n_samples)

    # 训练模型
    print('----- start training', config.model, '-----')
    if config.model in ['lstm', 'cnn1d', 'cnn2d']:
        y_train, y_val = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)  # 独热编码
        train_model(config, x_train, y_train, x_test, y_val, config.batch_size, config.epochs)
    else:
        train_model(x_train, y_train)
    print('----- end training ', config.model, ' -----')


if __name__ == '__main__':
    config = opts.parse_opt()
    train(config)

