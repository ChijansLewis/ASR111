#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: XIAOJING
用于测试语音识别系统语音模型的程序

"""
import os
import torch
import random
from readdata24 import DataSpeech
from SpeechModel251_pytorch import Net
from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *
from ctc_decoder import decode

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
AUDIO_LENGTH = 1600  # 16s的音频
AUDIO_FEATURE_LENGTH = 200  # 2s特征
batch_size = 1


def predict(model, data_input, input_len):
    """
    预测结果
    返回语音识别后的拼音符号列表
    """

    in_len = np.zeros(batch_size, dtype=np.int32)
    in_len[0] = input_len
    x_in = np.zeros((batch_size, AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
    for i in range(batch_size):
        x_in[i, 0:len(data_input)] = data_input

    if device.type == "cuda":
        x_in = torch.Tensor(x_in).cuda()
    else:
        x_in = torch.Tensor(x_in)

    base_pred = model(x_in)
    base_pred = torch.squeeze(base_pred)
    base_pred = base_pred.cpu().detach().numpy()
    labels, score = decode(base_pred)
    return labels, score


def test(net, data_path, str_dataset='test', data_count=128):
    data = DataSpeech(data_path, str_dataset)
    num_data = data.GetDataNum()  # 获取数据的数量
    if data_count <= 0 or data_count > num_data:  # 当data_count为<=0或者>测试数据量的值时，则使用全部数据来测试
        data_count = num_data
    ran_num = random.randint(0, num_data - 1)  # 获取一个随机数
    words_num = 0
    word_error_num = 0

    for i in range(data_count):
        data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据

        # 数据格式出错处理 开始
        # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
        num_bias = 0
        while data_input.shape[0] > AUDIO_LENGTH:
            print('*[Error]', 'wave data length of num', (ran_num + i) % num_data, 'is too long.',
                  '\n A Exception raise when test Speech Model.')
            num_bias += 1
            data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)  # 从随机数开始连续向后取一定数量数据
        # 数据格式出错处理 结束

        preds, score = predict(net, data_input, data_input.shape[0] // 8)

        words_n = data_labels.shape[0]  # 获取每个句子的字数
        words_num += words_n  # 把句子的总字数加上
        edit_distance = GetEditDistance(data_labels, preds)  # 获取编辑距离
        if edit_distance <= words_n:  # 当编辑距离小于等于句子字数时
            word_error_num += edit_distance  # 使用编辑距离作为错误字数
        else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
            word_error_num += words_n  # 就直接加句子本来的总字数就好了

        print('True:%s Pred:%s' % (str(data_labels), str(preds)))

    # print('*[Test Result] Speech Recognition ' + str_dataset + ' set word error ratio: ',
    #       word_error_num / words_num * 100, '%')
    print('[测试结果] 语音识别: %s 集语音单字错误率：%.2f' % (str_dataset, word_error_num / words_num))


def recognize_speech(net, wav_signal, fs):
    """
    最终做语音识别用的函数，识别一个wav序列的语音
    """

    # 获取输入特征
    # data_input = GetMfccFeature(wav_signal, fs)
    data_input = GetFrequencyFeature3(wav_signal, fs)

    input_length = len(data_input)
    input_length = input_length // 8

    data_input = np.array(data_input, dtype=np.float)
    data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
    preds, _ = predict(net, data_input, input_length)
    list_symbol_dic = GetSymbolList('./')  # 获取拼音列表

    r_str = []
    for i in preds:
        r_str.append(list_symbol_dic[i])

    return r_str


if __name__ == '__main__':
    data_path = '../dataset'
    model_path = './model_speech/300_7.4032.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    net = Net()
    # 加载已训练好的模型权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device).eval()
    # test(net, data_path, str_dataset='test', data_count=1)
    filename = r'../dataset/ST-CMDS-20170001_1-OS/20170001P00001A0001.wav'
    wav_signal, fs = read_wav_data(filename)
    r = recognize_speech(net, wav_signal, fs)
    print('语音识别结果：', r)
