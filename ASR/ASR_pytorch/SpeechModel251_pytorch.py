#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: xiaojing
@function: 定义net
"""

import torch
from torch import nn
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # backend
        self.pre_layer1 = nn.Sequential(
            # 第1层
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            # 第2层
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 第3层
            nn.MaxPool2d(kernel_size=2, stride=None),
            nn.Dropout(0.05),
            # 第4层
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 第5层
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 第6层
            nn.MaxPool2d(kernel_size=2, stride=None),
            nn.Dropout(0.1),
            # 第7层
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            # 第8层
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 第9层
            nn.MaxPool2d(kernel_size=2, stride=None),
            nn.Dropout(0.15),
            # 第10层
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 第11层
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 第12层
            nn.MaxPool2d(kernel_size=1, stride=None),
            nn.Dropout(0.2),
            # 第13层
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 第14层
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 第15层
            nn.MaxPool2d(kernel_size=1, stride=None),
            nn.Dropout(0.3),
        )
        self.fc = nn.Sequential(
            # 第16层
            nn.Linear(128 * 25, 128, bias=True),
            nn.Dropout(0.3),
            # 第17层
            nn.Linear(128, 1428, bias=True),  # 输出1428个参数值,即1427个拼音+1个空白块
        )

        # self.softmax = nn.Softmax(dim=2)  # loss输出全为负数，不可行
        # self.softmax = nn.LogSoftmax(dim=2)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # from NHWC to NCHW
        x = self.pre_layer1(x)
        x = x.reshape(x.size(0), x.size(2), -1)  # 4 200 3200
        x = self.fc(x)
        # x = F.log_softmax(x, dim=2)
        return x
