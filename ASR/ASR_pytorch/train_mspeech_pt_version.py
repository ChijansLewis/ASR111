#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: xiaojing
@function: 用于训练语音识别系统语音模型的程序
"""
import os
from SpeechModel251_pytorch import Net
from torch import optim
from torch import nn
import torch
from readdata24 import DataSpeech
from general_function.drawing import draw
from torch.nn import functional as F

model_path = 'model_speech'
data_path = '../dataset'
batch_size = 4
max_epoch = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
AUDIO_LENGTH = 1600  # 1.6s的音频
AUDIO_FEATURE_LENGTH = 200  # 0.2s特征


def train(save_path, device, batch_size=4):
    # 数据加载
    data_speech = DataSpeech(data_path, 'train')
    num_data = data_speech.GetDataNum()  # 获取数据的数量
    yield_datas = data_speech.data_genetator(batch_size, AUDIO_LENGTH)
    epoch_size = num_data // batch_size  # 每一训练轮次的迭代次数

    # 定义网络
    net = Net()
    net.to(device).train()  # 用gpu加载模型速度快
    pretrained = False
    if pretrained:
        net.load_state_dict(torch.load('./model_speech/1100_-3.2020.pt'))

    # 构建损失函数及优化器
    tmp_lr = 1e-3
    optimizer = optim.SGD(net.parameters(), lr=tmp_lr, momentum=0.9)
    # ctc_loss = nn.CTCLoss()
    ctc_loss = nn.CTCLoss(blank=1427, reduction='mean')
    # 定义训练过程
    loss_list, epoch_list = [], []
    for cur_epoch in range(max_epoch):
        # 使用阶梯学习率衰减策略
        # if cur_epoch in (0.6 * max_epoch, 0.8 * max_epoch):  # 在60%，80%都进行lr下降r
        if cur_epoch > 0:
            tmp_lr = tmp_lr * 0.99
            for param_group in optimizer.param_groups:
                param_group['lr'] = tmp_lr

        for iter_i, data in enumerate(yield_datas):
            if iter_i > epoch_size:
                break
            # 前向推理和计算损失
            y_pred = net(torch.Tensor(data[0][0]).to(device)).log_softmax(dim=2)  # format [T B C]
            labels = torch.Tensor(data[0][1]).to(device)
            input_length = torch.IntTensor(data[0][2]).to(device)
            label_length = torch.IntTensor(data[0][3]).to(device)
            y_pred = y_pred.permute(1, 0, 2)
            input_length = torch.squeeze(input_length)
            label_length = torch.squeeze(label_length)

            loss = ctc_loss(y_pred, labels, input_length, label_length)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 输出结果
            if iter_i % 1 == 0:
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]  Loss: %.6f'
                      % (cur_epoch + 1, max_epoch, iter_i, epoch_size, tmp_lr, loss.item()))
                loss_list.append(loss)
                epoch_list.append(iter_i)
                draw(epoch_list, loss_list)

            if iter_i % 100 == 0 and iter_i != 0:
                torch.save(net.state_dict(), "./%s/%d_%.4f.pt" % (save_path, iter_i, loss))
                # 保存模型的推理过程的时候，只需要保存模型训练好的参数，
                # 使用torch.save()保存state_dict，能够方便模型的加载

        # loss_list.append(loss)
        # epoch_list.append(cur_epoch)
        # draw(loss_list, epoch_list)


if __name__ == '__main__':

    # print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if not os.path.exists(model_path):  # 判断保存模型的目录是否存在
        os.makedirs(model_path)  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    train(model_path, device, batch_size)
