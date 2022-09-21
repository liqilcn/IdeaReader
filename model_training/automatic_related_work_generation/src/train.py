#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
import config
from config import BertSumAbs_Train_Args, BART_Train_Args
from others.logging import init_logger
from train_bertsumabs import train_bertsumabs, validate_bertsumabs, test_bertsumabs, test_text_bertsumabs
from train_bart import train_bart, validate_bart

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    target_model = config.target_model
    if (target_model == 'bertsumabs'):
        args = BertSumAbs_Train_Args()
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]  # 多卡训练的时候用于训练的GPU编号，是一个整数列表，重新编号，从0开始
        args.world_size = len(args.gpu_ranks)  # 多卡训练时的GPU数量，即全局的进程个数
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus  # 这是实际的GPU

        init_logger(args.log_file)  # 在调用其他py文件的文件中初始化logger，其他被调用的文件也可以使用，不必再可视化
        device = "cpu" if args.visible_gpus == '-1' else "cuda"
        device_id = 0 if device == "cuda" else -1  # 如果用于gpu训练，则device_id为0，否则为-1

        if (args.mode == 'train'):
            train_bertsumabs(args, device_id)
        elif (args.mode == 'validate'):
            validate_bertsumabs(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_bertsumabs(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_bertsumabs(args, device_id, cp, step)

    elif (target_model == 'bart'):
        args = BART_Train_Args()
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]  # 多卡训练的时候用于训练的GPU编号，是一个整数列表，重新编号，从0开始
        args.world_size = len(args.gpu_ranks)  # 多卡训练时的GPU数量，即全局的进程个数
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus  # 这是实际的GPU

        init_logger(args.log_file)
        device = "cpu" if args.visible_gpus == '-1' else "cuda"
        device_id = 0 if device == "cuda" else -1  # 如果用于gpu训练，则device_id为0，否则为-1
        if (args.mode == 'train'):
            train_bart(args, device_id)
        elif (args.mode == 'validate'):
            validate_bart(args, device_id)
        # 其他模型如上
        pass
    elif (target_model == 'other_target_model2'):
        pass
