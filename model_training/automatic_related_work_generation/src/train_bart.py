#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import json
import os
import random
import signal
import time

import torch

import distributed
from models.data_loader_bart import SurveyGenDataset
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from models.optimizers import Optimizer
from models.predictor_bart import bart_generator
from models.trainer_bart import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_bart_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size  # nb_gpu=args.world_size, 即GPU个数，pytorch并行是一个GPU上一个进程
    mp = torch.multiprocessing.get_context('spawn')  # 用spawn方式启动，get_context方法为获取multiprocessing的上下文，即初始化multiprocessing对象并使其启动方式为spawn
    # get_context方法类似set_start_method，但略有不同
    # torch.multiprocessing.get_context的操作是Python multiprocessing的替代品。它支持完全相同的操作，但扩展了它

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()  # join：等待子进程结束后，父进程才继续执行


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        # 该函数基于torch.distributed.init_process_group，
        # torch.distributed.init_process_group需要在每个进程中进行调用，用于初始化该进程。在使用分布式时，该函数必须在 distributed 内所有相关函数之前使用。
        # 通用套路，扩展框架无需更改
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)  # 每个进程在使用前必须调用该函数进行初始化
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:  # 如果返回的gpu_rank和分配的rank不同，则报错退出
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_bart_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)

def build_optim_bart(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:  # 不同checkpoint的学习率是不同的，所以需要把优化器中学习率等参数存储在checkpoint中
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',  # 变换学习率的方法
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def train_bart(args, device_id):
    if (args.world_size > 1):
        train_bart_multi(args)
    else:
        train_bart_single(args, device_id)


def train_bart_single(args, device_id):
    # device_id：多进程时device_id为从0开始的整数，与GPU，或者进程数相对应
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)  # 确定随机种子，使得程序在每次运行时产生的随机数都相同
    random.seed(args.seed)  
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        # 在多GPU情况下，用于设置当前进程所使用的GPU
        # 将前面os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus中第device_id设置为当前使用的GPU
        # device_id不管单进程还是多进程，总是从0开始
        # 前面os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus设置了具体使用的GPU设备，
        # torch.cuda.set_device是将前面的GPU设备从0开始重新编号的
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)  # 将checkpoint加载到GPU上，lambda storage, loc: storage，这个lambda函数就是将参数加载到GPU上
        opt = vars(checkpoint['opt']) # vars() 函数返回对象object的属性和属性值的字典对象
        for k in opt.keys():  # 加载args
            if (k in model_flags):
                setattr(args, k, opt[k])  
    else:
        checkpoint = None
    # 前面是加载checkpoint的预处理阶段，如果有checkpoint，就加载checkpoint中的模型的超参数，并初始化args对象，而忽略命令行传入的超参数，若不存在checkpoint，则使用命令行的超参数训练

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():  # dataloader迭代器
        return DataLoader(dataset=SurveyGenDataset(os.path.join(args.torch_data_path, f'{args.mode}.pt')),
                          batch_size=args.batch_size, shuffle=True)
    model_name = "facebook/bart-large" if args.large else "facebook/bart-base"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    if args.train_from != '':
        model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    optimizer = build_optim_bart(args, model, checkpoint)

    logger.info(model)
    trainer = build_trainer(args, device_id, model, optimizer)
    # trainer是通用的，可以针对不同的模型，这样又提高了代码的通用性
    trainer.train(train_iter_fct, args.train_steps)

def validate_bart(args, device_id):
    # 先用交叉熵loss选出5个最优模型，然后使用这个5个模型最后对测试集进行测试，生成最终的文本以及reference，选择最优的作为最后的模型
    # batch_size和训练的batch_size一样
    timestep = 0
    if (args.test_all):  # 如果要对所有的checkpoint进行测试
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))  # cp(checkpoint)
        cp_files.sort(key=os.path.getmtime)  # os.path.getmtime: 获取指定路径的最后修改时间
        xent_lst = []  # xent：cross entropy
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            # 下面这个if是排除test_start_from之前的cp
            if (args.test_start_from != -1 and step < args.test_start_from):  # 从第args.test_start_from个checkpoint之后进行验证，args.test_start_from之前的checkpoint赋予很大的xent
                xent_lst.append((1e6, cp))  # cp对应的checkpoint文件赋予很大的xent
                continue
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            # max_step = xent_lst.index(min(xent_lst))  # 取当前xent值最小的cp的step作为max step
            # if (i - max_step > 10):  # 若当前step的index i 与max step相距超过10，则退出validate
            #     break
        # 总之前面就是遍历所有的checkpoint（cp），以xent作为每个cp表现的指标，得到一个大的candidate
        xent_lst = sorted(xent_lst, key=lambda x: x[0])
        json.dump(xent_lst, open(os.path.join(args.result_path, f'cp_xents.json'), 'w'))
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst[:5]: # 然后再在xent最小的前5个checkpoint中分别再用test数据集进行测试
            step = int(cp.split('.')[-2].split('_')[-1])  # step: checkpoint对应的训练步数
            test_bart(args, device_id, cp, step)  # 对test数据集进行测试，会按照步骤将测试的结果存储在3个文件中
    else:  # 在该模式下，当训练正在进行的时候，validate与test也同样进行，即总是完成对最新的checkpoint进行验证
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)  # 当前cp的最后修改时间
                if (not os.path.getsize(cp) > 0):  # 如果cp的大小为0，则程序睡眠60s
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):  # 若文件大小不为0，且当前cp的最后修改时间大于最新的cp的修改时间，则进行验证与test
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_bart(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):  # 验证相应的checkpoint的performance
    # validate的时候不需要优化器
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    model_name = "facebook/bart-large" if args.large else "facebook/bart-base"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(checkpoint['model'], strict=True)

    model.to(device)
    model.eval()

    valid_iter = DataLoader(dataset=SurveyGenDataset(os.path.join(args.torch_data_path, f'{args.mode}.pt')),
                          batch_size=args.batch_size, shuffle=True)

    trainer = build_trainer(args, device_id, model, None)  # 优化器为None
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test_bart(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    model_name = "facebook/bart-large" if args.large else "facebook/bart-base"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    model.eval()

    test_iter = DataLoader(dataset=SurveyGenDataset(os.path.join(args.torch_data_path, 'test.pt')),
                          batch_size=args.test_batch_size, shuffle=True)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    bart_generator(args, test_iter, model, tokenizer, step)