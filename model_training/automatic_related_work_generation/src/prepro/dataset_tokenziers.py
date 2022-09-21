#!/usr/bin/env python
# coding: utf-8

# ### 新增数据集时只需要改写这个py文件，新增一个相应数据集的tokenzier，该py通过读取config中的相应数据集，将处理完成的对应数据集返回统一的token格式

# In[1]:


import os
import nltk
import nltk.tokenize
from nltk import data
data.path.append('./nltk_data')

# nltk会首先下载用于分词与分句的包
# nltk.download('stopwords')
# nltk.download('punkt')
import sys
import json
import config
from tqdm import tqdm


# In[2]:


def delve_tokenizer():
    # 输入的任意格式数据集的结构，划分为'train', 'valid', 'test'，每个集合存储为1个列表，每个列表元素为一个数据实例，其中包含多文档的分词以及多文档总结的分词
    dataset = ['train', 'valid', 'test']
    tokenize_dataset = {}
    for ds in dataset:
        with open(os.path.join(config.raw_dataset_path, f'{ds}.json'), 'r') as fp:
            print(f'tokenize {ds}...')
            jsonlines = fp.readlines()
            final_data_list = []
            for line in tqdm(jsonlines):
                json_line = json.loads(line)
                muti_doc = json_line['multi_doc']
                mds_sum = json_line['abs']
                muti_doc_tokenize = []
                # 对多文档列表进行分词
                for doc in muti_doc:
                    sent_tokenize = nltk.tokenize.sent_tokenize(doc)
                    sent_tokenize_list = []
                    for sent in sent_tokenize:
                        sent_tokenize_list.append(nltk.tokenize.word_tokenize(sent))
                    muti_doc_tokenize.append(sent_tokenize_list)
                # 对多文档总结进行分句，然后对单个句子进行分词
                mds_sum_sent_tokenize = nltk.tokenize.sent_tokenize(mds_sum)
                mds_sum_tokenize = []
                for sent in mds_sum_sent_tokenize:
                    mds_sum_tokenize.append(nltk.tokenize.word_tokenize(sent))
                final_data_list.append({'multi_doc':muti_doc_tokenize, 'abs': mds_sum_tokenize})
        tokenize_dataset[ds] = final_data_list
    return tokenize_dataset


# In[5]:


def s2orc_tokenizer():
    pass


# In[6]:


def dataset_tokenize():
     return eval(f'{config.preprocess_dataset}_tokenizer()')
