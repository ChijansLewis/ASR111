#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from SpeechModel251 import ModelSpeech

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 进行配置，使用90%的GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

if __name__ == '__main__':
    datapath = '../dataset'
    ms = ModelSpeech(datapath)
    ms.LoadModel('./model_speech/m251/speech_model251_e_0_step_12400.base.h5')
    # ms.TestModel(datapath, str_dataset='test', data_count=128, out_report=True)
    r = ms.RecognizeSpeech_FromFile(r'..\dataset\ST-CMDS-20170001_1-OS\20170001P00001A0002.wav')
    print('语音识别结果：', r)
