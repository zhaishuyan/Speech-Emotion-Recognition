# from keras.layers import LSTM as KERAS_LSTM
# from keras.layers import Dense, Dropout
import torch
import torch.nn as nn
import numpy as np
from .dnn import DNN_Model


# LSTM
class LSTM(DNN_Model):
    def __init__(self, rnn_size, dropout=0.5, **params):
        params['name'] = 'LSTM'
        super(LSTM, self).__init__(**params)
        self.lstm = nn.LSTM((1 * self.input_shape), rnn_size)
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(rnn_size, self.hidden_size)
        self.relu = nn.ReLU(self.hidden_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        dropout_out = self.dropout(lstm_out)
        l1_out = self.l1(dropout_out)
        relu_out = self.relu(l1_out)
        l2_out = self.l2(relu_out)
        # l2_out = self.l2(input)
        # softmax_out = self.softmax(l2_out)
        return l2_out

    def reshape_input(self, data):
        # 二维数组转三维
        # (n_samples, n_feats) -> (n_samples, time_steps = 1, input_size = n_feats)
        # time_steps * input_size = n_feats
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        return data

    def get_model(self):
        self.model = LSTM()

    '''
    搭建 LSTM

    输入:
        rnn_size(int): LSTM 隐藏层大小
        hidden_size(int): 全连接层大小
        dropout(float)
    '''

    # def make_model(self, rnn_size, hidden_size, dropout=0.5, **params):
    #     self.model.add_module('lstm', nn.LSTM((1 * self.input_shape), rnn_size))  # (time_steps = 1, n_feats)
    #     self.model.add_module('dropout', nn.Dropout(dropout))
    #     self.model.add_module('l1', nn.Linear(rnn_size, hidden_size))
    #     self.model.add_module('relu', nn.ReLU(hidden_size))
    #     print('hidden_size: ', hidden_size)
    #     print('rnn_size: ', rnn_size)
    #     self.hidden_size = hidden_size
    #     # self.model.add(Dense(rnn_size, activation='tanh'))
