# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging

# 此模块无需更改，直接调用即可

logger = logging.getLogger()  # 其他文件要使用logger的时候会将logger，init_logger导入，就相当于自己定义了个logger，然后直接调用init_logger对这个logger进行初始化，init_logger不需要接返回值


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")  # 设置日志的输出格式
    logger = logging.getLogger()  # 定义一个logger
    logger.setLevel(logging.INFO)  # 设置这个logger的输出level

    console_handler = logging.StreamHandler()  # 定义一个将日志输出到命令行的Handler，Handler用于将日志输出到不同的位置如文件和命令行，可以自己定义输出的格式以及level等
    console_handler.setFormatter(log_format)  # 输出日志的格式
    logger.handlers = [console_handler]  # logger的handlers

    if log_file and log_file != '':  # 如果配置文件存在log_file,则定义一个将日志输出到文件的handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
