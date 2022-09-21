# 读懂后直接用来作为自己的dataloader
import bisect
import os
import gc
import glob
import random

import torch

from others.logging import logger



class Batch(object):
    # Batch能够自适应改变batch大小
    def _pad(self, data, pad_id, width=-1):
        # 将数据长度用pad填充成等长，填充后的最终数据长度是这个batch中最长数据的长度，因此每个batch中example的长也不尽相等，BERT词表中ID为0的token就是pad
        if (width == -1):
            width = max(len(d) for d in data)  # 最终数据的长度，不同的batch中的width不同
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]  # 在后面填充pad
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None: # 此处处理为将截断后的数据添加PAD与MASK, PAD即对相关序列补全0，相关序列的MASK即为True和False列表
            # 相比阶段后的数据，此处数据在自身增加了PAD, 并增加了相应的mask字段
            self.batch_size = len(data)  # 实际的batch size在这，而不是初始命令行输入的batch_size数值，batch size根据数据的toke长度自适应调节，以提高内存使用效率
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_src_sent_labels = [x[4] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))

            segs = torch.tensor(self._pad(pre_segs, 0))
            # mask_src = 1 - (src == 0)
            # mask_tgt = 1 - (tgt == 0)
            mask_src = ~(src == 0)  # token中，bert词表中id为0的token为[PAD]
            mask_tgt = ~(tgt == 0)
            # 此处新版本torch已经不再支持1-
            # 改为：~(torch.tensor([1,2,3,4,5,6,0,0,0,0])==0)
            # 结果为：tensor([ True,  True,  True,  True,  True,  True, False, False, False, False])  # example的一个mask，确定哪部分是真实数据，哪部分是pad

            clss = torch.tensor(self._pad(pre_clss, -1))  # clss是位置，列表中已经存在0的，即从0开始，所以pad的标志用-1代替
            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
            # mask_cls = 1 - (clss == -1)
            mask_cls = ~ (clss == -1)  # mask是0-1矩阵
            clss[clss == -1] = 0  # 知道mask之后，可以将-1改成0，此时的clss应该为clss[0]=0, 后几个元素也为0
            # clss的几步转换如下：
            """
            ====================================================================================================
            pre_clss: [[0, 33, 51, 85, 131, 151, 188, 206, 231, 256, 283, 330, 361, 385, 416, 441, 466], [0, 83, 129, 151, 180, 232, 286, 315, 353, 390, 426, 450], [0, 17, 30, 60, 87, 110, 130, 155, 180, 217, 240, 284, 309, 330, 353, 369, 424, 454, 481]]
            clss = torch.tensor(self._pad(pre_clss, -1))操作: 
            tensor([[  0,  33,  51,  85, 131, 151, 188, 206, 231, 256, 283, 330, 361, 385,
                     416, 441, 466,  -1,  -1],
                    [  0,  83, 129, 151, 180, 232, 286, 315, 353, 390, 426, 450,  -1,  -1,
                      -1,  -1,  -1,  -1,  -1],
                    [  0,  17,  30,  60,  87, 110, 130, 155, 180, 217, 240, 284, 309, 330,
                     353, 369, 424, 454, 481]])
            mask_cls = ~ (clss == -1)操作：
            tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                      True,  True,  True,  True,  True,  True,  True, False, False],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                      True,  True, False, False, False, False, False, False, False],
                    [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                      True,  True,  True,  True,  True,  True,  True,  True,  True]])
            clss[clss == -1] = 0操作：
            tensor([[  0,  33,  51,  85, 131, 151, 188, 206, 231, 256, 283, 330, 361, 385,
                     416, 441, 466,   0,   0],
                    [  0,  83, 129, 151, 180, 232, 286, 315, 353, 390, 426, 450,   0,   0,
                       0,   0,   0,   0,   0],
                    [  0,  17,  30,  60,  87, 110, 130, 155, 180, 217, 240, 284, 309, 330,
                     353, 369, 424, 454, 481]])
            ====================================================================================================

            """
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))


            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))


            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size




def load_dataset(args, corpus_type, shuffle):
    # 将数据集切片并采用yield的方式能够节约内存
    # 一般不需要更改
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(os.path.join(args.torch_data_path, corpus_type + '.[0-9]*.pt')))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!  # 这个else是数据集未切片的情况，即只有一个corpus_type.pt文件
        pt = os.path.join(args.torch_data_path, corpus_type + '.pt')
        yield _lazy_dataset_loader(pt, corpus_type)


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size  # src_elements实际是tgt按照当前数据中最大长度计算时，minibatch列表（在调用函数中）中token的数量，即当前minibatch列表中所有tgt token的一个估计的上界，max_size是实际数据当中tgt的最大长度
    if (count > 6):
        return src_elements + 1e3  # 由于实际最终的mini batch的size是个位数，加一个1e3是一个惩罚项，为了当实际数据中的tgt较短时，限制mini batch的size仍然保持较短的长度
    return src_elements


class Dataloader(object):
    # 最终Dataloader吐出的batch长度不是一个固定的长度，会随数据动态调整，这个有别于pytorch提供的dataloader，并且功能更强大
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)  # 先对self.cur_iter用self._next_dataset_iterator(datasets)进行初始化，但调用self.cur_iter之后不会再调用self._next_dataset_iterator(datasets)函数初始化
        assert self.cur_iter is not None

    def __iter__(self):  # 遍历Dataloader的时候会调用__iter__(self)函数，__iter__函数代表Dataloader类可以进行迭代，每次迭代返回一个batch
        # 对dataloader进行迭代的时候每次返回一个batch
        dataset_iter = (d for d in self.datasets)  # ()定义生成器
        while self.cur_iter is not None: # 这一层是对所有的pt文件进行遍历
            for batch in self.cur_iter:  # 这一层是返回一个batch，self.cur_iter追溯到头是一个class DataIterator(object)对象，遍历DataIterator对象的时候会调用其自身的__iter__(self)函数
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()  
                del self.cur_dataset
                gc.collect()  # python对内存垃圾进行回收

            self.cur_dataset = next(dataset_iter)  # 获取下一个数据集切片的pt文件
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

        self.batch_size_fn = abs_batch_size_fn

    def data(self):
        # 打乱数据集，但由于随机种子是一样的，所以每次运行程序的数据集顺序是一样的
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs






    def preprocess(self, ex, is_test):
        # 根据命令行参数对preprocess处理完的pt文件中各个元素再进行一次截断处理，实际格式并无变化
        src = ex['src']
        # 2在bert的词表中对用token [unused1]，前面预处理也是直接用[unused1]结尾，这里就默认知道
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1]+[2]  # 如果tgt长度大于max_tgt_len，将tgt在max_tgt_len-1的地方截断，在最后一个位置插入2，使得包括2整个tgt的长度为self.args.max_tgt_len，否则保持原样
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id  # 如果src长度大于args.max_pos，将src在max_pos-1的地方截断，在最后一个位置插入结束标志，使得包括结束token整个tgt的长度为self.args.max_tgt_len，否则保持原样
        # end_id对应的token为[SEP]，由于不好数，故反向索引，与tgt加2方式不同，但本质相同
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)  # 在设置args.max_pos之后，文本截断之后实际保留的句子的个数，从而确定最终输入模型的src_sent_labels和clss
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]



        if(is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    # 上面函数的实际数据转换如下
    """
    输入：{'src': [101, 1031, 1014, 1033, 2057, 5136, 1996, 3291, 1997, 23208, 1037, 2312, 2193, 1997, 26726, 8988, 11423, 2006, 2019, 20950, 5460, 1012, 102, 101, 2256, 2364, 6691, 3774, 1999, 4760, 2008, 28283, 25300, 10074, 10713, 8285, 21022, 1006, 1040, 7011, 1007, 2064, 2022, 2109, 6464, 2005, 2023, 3291, 1024, 1999, 2256, 7885, 2057, 6162, 1037, 2083, 18780, 1997, 2055, 1019, 29625, 2549, 29232, 1010, 2981, 1997, 1996, 2193, 1997, 26726, 8988, 11423, 1006, 2039, 2000, 1015, 29623, 8889, 2692, 29623, 8889, 2692, 1999, 2256, 5852, 1007, 1012, 102, 101, 1996, 2350, 3291, 2057, 2227, 2003, 2008, 1997, 1996, 2946, 1997, 1996, 1040, 7011, 1012, 102, 101, 2144, 1996, 2193, 1997, 2163, 7502, 27258, 2135, 2007, 1996, 2193, 1997, 26726, 8988, 11423, 1010, 2009, 2001, 3130, 3373, 2008, 1040, 7011, 2015, 2064, 2025, 2022, 2109, 2000, 2832, 2312, 4520, 1997, 11423, 1012, 102, 101, 2057, 2191, 1037, 9373, 4106, 1997, 1996, 2193, 1997, 2163, 1999, 1996, 1040, 7011, 4525, 2013, 26726, 8988, 11423, 1010, 1998, 5136, 2119, 1996, 2553, 2043, 2009, 2003, 3833, 17858, 1010, 1998, 2043, 2009, 2003, 3833, 2474, 28431, 1012, 102, 101, 2256, 4106, 7127, 2008, 1010, 2043, 1996, 8285, 18900, 2239, 2003, 3833, 2474, 28431, 1010, 1998, 2104, 3056, 17568, 2055, 1996, 3252, 1997, 1996, 7953, 20950, 2951, 1010, 1996, 2193, 1997, 2163, 1999, 1996, 13971, 1040, 7011, 2003, 6133, 3085, 1012, 102, 101, 2057, 2036, 9398, 3686, 6388, 2135, 2256, 9556, 1010, 2006, 2119, 12553, 1998, 2613, 20950, 2951, 4520, 1012, 102, 101, 1031, 1015, 1033, 2592, 28170, 5097, 2024, 8550, 4852, 6217, 2349, 2000, 6918, 8377, 1999, 4806, 20235, 1998, 1057, 5638, 15549, 3723, 1012, 102, 101, 1996, 11591, 3872, 1997, 2951, 2800, 26785, 7971, 17570, 2015, 1996, 2224, 1997, 13228, 8107, 2000, 28170, 1999, 2344, 2000, 4468, 10827, 5198, 2007, 14203, 2592, 1012, 102, 101, 4493, 10595, 2005, 13228, 28170, 4050, 11160, 2006, 3722, 3145, 18351, 9844, 2030, 4524, 1997, 2616, 2592, 26384, 5461, 1012, 102, 101, 1996, 13896, 1997, 20950, 2004, 1037, 3115, 2005, 2592, 3863, 1998, 1996, 2458, 1997, 23032, 4155, 2005, 20950, 2951, 12939, 1996, 2458, 1997, 2062, 12138, 22910, 10595, 2008, 2202, 3252, 2592, 2046, 4070, 1012, 102, 101, 2057, 2031, 2764, 2195, 5950, 4411, 1998, 3945, 13792, 2005, 4488, 8114, 22910, 1997, 20950, 5491, 2005, 2312, 15782, 2571, 2592, 28170, 3001, 1012, 102, 101, 1999, 2023, 3259, 2057, 6235, 2122, 5461, 1998, 11628, 2037, 2836, 2408, 1037, 2846, 1997, 6254, 1010, 2147, 11066, 1010, 1998, 4094, 16820, 1012, 102, 101, 1031, 1016, 1033, 2574, 1010, 2172, 1997, 1996, 2951, 10573, 2058, 1996, 4274, 2097, 2022, 12359, 1999, 20950, 1010, 4352, 2005, 12138, 22910, 1998, 4180, 15058, 2094, 16972, 1012, 102, 101, 2057, 2031, 2328, 1037, 22910, 3194, 2170, 1061, 8873, 21928, 1010, 2029, 17736, 11058, 20950, 5491, 2429, 2000, 1060, 4226, 2854, 2030, 26726, 8988, 10861, 5134, 2008, 9125, 2119, 4130, 11423, 1998, 3653, 16467, 2015, 1012, 102, 101, 4406, 3025, 2147, 1010, 1061, 8873, 21928, 3594, 1037, 3117, 1050, 7011, 15058, 2094, 7781, 2944, 1012, 102, 101, 1999, 2023, 10467, 1010, 2057, 2556, 1996, 5090, 1998, 13792, 10318, 1061, 8873, 21928, 1010, 1998, 2265, 2049, 8122, 1998, 26743, 8553, 2104, 2536, 2147, 11066, 2015, 1012, 102, 101, 1031, 1017, 1033, 2057, 16599, 1037, 3117, 5950, 3252, 1010, 12061, 1060, 18886, 2063, 1010, 2008, 6753, 1996, 8114, 22910, 1997, 20950, 5491, 2241, 2006, 26726, 8988, 11423, 1012, 102, 101, 2256, 1060, 18886, 2063, 5950, 3252, 4107, 2195, 3117, 2838, 2008, 2191, 2009, 2926, 8702, 2005, 2312, 15782, 2571, 10172, 6342, 5910, 26775, 20755, 3001, 1012, 102, 101, 2034, 1010, 1060, 18886, 2063, 2003, 2881, 2000, 2490, 4621, 22910, 2241, 2006, 3375, 26726, 8988, 11423, 1006, 2004, 4941, 2000, 3722, 1010, 2309, 15069, 15480, 1007, 1012, 102, 101, 2117, 1010, 2256, 1060, 18886, 2063, 3252, 1998, 13792, 2024, 2881, 2000, 2490, 2119, 3641, 1998, 27776, 26764, 2098, 9844, 1997, 20950, 2951, 1012, 102, 101, 2353, 1010, 2011, 5950, 2075, 2006, 10071, 1997, 5783, 3415, 4114, 1999, 1037, 13012, 2063, 3252, 1998, 2478, 1037, 12138, 9844, 9896, 1010, 1060, 18886, 2063, 2003, 2583, 2000, 2119, 5547, 1996, 2193, 1997, 14203, 5950, 15113, 2015, 2004, 2092, 2004, 4468, 21707, 9844, 2015, 1010, 8558, 4346, 5186, 8114, 22910, 1012, 102, 101, 2256, 6388, 3463, 2058, 1037, 2898, 2846, 1997, 20950, 6254, 1998, 26726, 8988, 3670, 2147, 11066, 2015, 10580, 2008, 2256, 1060, 18886, 2063, 5950, 3252, 2041, 4842, 22694, 3041, 8107, 2011, 2898, 17034, 1012, 102], 'tgt': [1, 13971, 20952, 2050, 1031, 1014, 1033, 1010, 1998, 2060, 2512, 1011, 28283, 25300, 10074, 10713, 8285, 21022, 1006, 1050, 7011, 1007, 2241, 13792, 1006, 1060, 8873, 21928, 1031, 1015, 1033, 1010, 1061, 8873, 21928, 1031, 1016, 1033, 1010, 1998, 1060, 18886, 2063, 1031, 1017, 1033, 1007, 2031, 2042, 3818, 2004, 8114, 13792, 2005, 6364, 1037, 2312, 2193, 1997, 26726, 2229, 1999, 20950, 9199, 1012, 3, 13971, 20952, 2050, 1010, 2029, 2474, 28431, 1036, 1036, 9570, 2015, 28283, 25300, 10074, 10713, 8285, 21022, 1006, 1040, 7011, 1007, 1010, 2003, 6020, 2000, 1996, 2500, 1999, 3408, 1997, 6364, 2836, 1010, 2138, 2009, 16021, 14900, 1037, 5377, 2152, 2083, 18780, 1006, 2004, 3435, 2004, 2019, 20950, 11968, 8043, 1007, 2005, 1037, 3074, 1997, 2309, 1011, 4130, 26726, 2229, 1010, 2096, 1996, 2500, 1005, 2836, 2003, 7399, 2135, 26131, 2114, 1996, 2193, 1997, 26726, 2229, 1012, 3, 2174, 1010, 2045, 2024, 2048, 18636, 3314, 4953, 13971, 20952, 2050, 1025, 1015, 1007, 2043, 2009, 6194, 3375, 20950, 5491, 1010, 2009, 5942, 1037, 2312, 2193, 1997, 1040, 7011, 2163, 1010, 1998, 2947, 2064, 2448, 2041, 1997, 3638, 1010, 1998, 1016, 1007, 2009, 2515, 1050, 1005, 1056, 5047, 23346, 26726, 2229, 1025, 2009, 3727, 2068, 2000, 1996, 5097, 1012, 2], 'src_sent_labels': [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'segs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clss': [0, 23, 88, 105, 142, 183, 226, 246, 271, 300, 322, 358, 384, 410, 441, 479, 498, 528, 559, 587, 617, 643, 697], 'src_txt': ['[ 0 ] we consider the problem of evaluating a large number of xpath expressions on an xml stream .', 'our main contribution consists in showing that deterministic finite automata ( dfa ) can be used effectively for this problem : in our experiments we achieve a throughput of about 5.4mbs , independent of the number of xpath expressions ( up to 1,000,000 in our tests ) .', 'the major problem we face is that of the size of the dfa .', 'since the number of states grows exponentially with the number of xpath expressions , it was previously believed that dfas can not be used to process large sets of expressions .', 'we make a theoretical analysis of the number of states in the dfa resulting from xpath expressions , and consider both the case when it is constructed eagerly , and when it is constructed lazily .', 'our analysis indicates that , when the automaton is constructed lazily , and under certain assumptions about the structure of the input xml data , the number of states in the lazy dfa is manageable .', 'we also validate experimentally our findings , on both synthetic and real xml data sets .', '[ 1 ] information dissemination applications are gaining increasing popularity due to dramatic improvements in communications bandwidth and ubiquity .', 'the sheer volume of data available necessitates the use of selective approaches to dissemination in order to avoid overwhelming users with unnecessary information .', 'existing mechanisms for selective dissemination typically rely on simple keyword matching or bag of words information retrieval techniques .', 'the advent of xml as a standard for information exchange and the development of query languages for xml data enables the development of more sophisticated filtering mechanisms that take structure information into account .', 'we have developed several index organizations and search algorithms for performing efficient filtering of xml documents for largescale information dissemination systems .', 'in this paper we describe these techniques and examine their performance across a range of document , workload , and scale scenarios .', '[ 2 ] soon , much of the data exchanged over the internet will be encoded in xml , allowing for sophisticated filtering and contentbased routing .', 'we have built a filtering engine called yfilter , which filters streaming xml documents according to xquery or xpath queries that involve both path expressions and predicates .', 'unlike previous work , yfilter uses a novel nfabased execution model .', 'in this demonstration , we present the structures and algorithms underlying yfilter , and show its efficiency and scalability under various workloads .', '[ 3 ] we propose a novel index structure , termed xtrie , that supports the efficient filtering of xml documents based on xpath expressions .', 'our xtrie index structure offers several novel features that make it especially attractive for largescale publishsubscribe systems .', 'first , xtrie is designed to support effective filtering based on complex xpath expressions ( as opposed to simple , singlepath specifications ) .', 'second , our xtrie structure and algorithms are designed to support both ordered and unordered matching of xml data .', 'third , by indexing on sequences of element names organized in a trie structure and using a sophisticated matching algorithm , xtrie is able to both reduce the number of unnecessary index probes as well as avoid redundant matchings , thereby providing extremely efficient filtering .', 'our experimental results over a wide range of xml document and xpath expression workloads demonstrate that our xtrie index structure outperforms earlier approaches by wide margins .'], 'tgt_txt': "lazydfa [ 0 ] , and other non-deterministic finite automata ( nfa ) based algorithms ( xfilter [ 1 ] , yfilter [ 2 ] , and xtrie [ 3 ] ) have been proposed as efficient algorithms for processing a large number of xpes in xml streams .<q>lazydfa , which lazily `` constructs deterministic finite automata ( dfa ) , is superior to the others in terms of processing performance , because it insures a constant high throughput ( as fast as an xml parser ) for a collection of single-path xpes , while the others ' performance is linearly degraded against the number of xpes .<q>however , there are two problematic issues regarding lazydfa ; 1 ) when it processes complex xml documents , it requires a large number of dfa states , and thus can run out of memory , and 2 ) it does n't handle branching xpes ; it leaves them to the applications ."}
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    输出：([101, 1031, 1014, 1033, 2057, 5136, 1996, 3291, 1997, 23208, 1037, 2312, 2193, 1997, 26726, 8988, 11423, 2006, 2019, 20950, 5460, 1012, 102, 101, 2256, 2364, 6691, 3774, 1999, 4760, 2008, 28283, 25300, 10074, 10713, 8285, 21022, 1006, 1040, 7011, 1007, 2064, 2022, 2109, 6464, 2005, 2023, 3291, 1024, 1999, 2256, 7885, 2057, 6162, 1037, 2083, 18780, 1997, 2055, 1019, 29625, 2549, 29232, 1010, 2981, 1997, 1996, 2193, 1997, 26726, 8988, 11423, 1006, 2039, 2000, 1015, 29623, 8889, 2692, 29623, 8889, 2692, 1999, 2256, 5852, 1007, 1012, 102, 101, 1996, 2350, 3291, 2057, 2227, 2003, 2008, 1997, 1996, 2946, 1997, 1996, 1040, 7011, 1012, 102, 101, 2144, 1996, 2193, 1997, 2163, 7502, 27258, 2135, 2007, 1996, 2193, 1997, 26726, 8988, 11423, 1010, 2009, 2001, 3130, 3373, 2008, 1040, 7011, 2015, 2064, 2025, 2022, 2109, 2000, 2832, 2312, 4520, 1997, 11423, 1012, 102, 101, 2057, 2191, 1037, 9373, 4106, 1997, 1996, 2193, 1997, 2163, 1999, 1996, 1040, 7011, 4525, 2013, 26726, 8988, 11423, 1010, 1998, 5136, 2119, 1996, 2553, 2043, 2009, 2003, 3833, 17858, 1010, 1998, 2043, 2009, 2003, 3833, 2474, 28431, 1012, 102, 101, 2256, 4106, 7127, 2008, 1010, 2043, 1996, 8285, 18900, 2239, 2003, 3833, 2474, 28431, 1010, 1998, 2104, 3056, 17568, 2055, 1996, 3252, 1997, 1996, 7953, 20950, 2951, 1010, 1996, 2193, 1997, 2163, 1999, 1996, 13971, 1040, 7011, 2003, 6133, 3085, 1012, 102, 101, 2057, 2036, 9398, 3686, 6388, 2135, 2256, 9556, 1010, 2006, 2119, 12553, 1998, 2613, 20950, 2951, 4520, 1012, 102, 101, 1031, 1015, 1033, 2592, 28170, 5097, 2024, 8550, 4852, 6217, 2349, 2000, 6918, 8377, 1999, 4806, 20235, 1998, 1057, 5638, 15549, 3723, 1012, 102, 101, 1996, 11591, 3872, 1997, 2951, 2800, 26785, 7971, 17570, 2015, 1996, 2224, 1997, 13228, 8107, 2000, 28170, 1999, 2344, 2000, 4468, 10827, 5198, 2007, 14203, 2592, 1012, 102, 101, 4493, 10595, 2005, 13228, 28170, 4050, 11160, 2006, 3722, 3145, 18351, 9844, 2030, 4524, 1997, 2616, 2592, 26384, 5461, 1012, 102, 101, 1996, 13896, 1997, 20950, 2004, 1037, 3115, 2005, 2592, 3863, 1998, 1996, 2458, 1997, 23032, 4155, 2005, 20950, 2951, 12939, 1996, 2458, 1997, 2062, 12138, 22910, 10595, 2008, 2202, 3252, 2592, 2046, 4070, 1012, 102, 101, 2057, 2031, 2764, 2195, 5950, 4411, 1998, 3945, 13792, 2005, 4488, 8114, 22910, 1997, 20950, 5491, 2005, 2312, 15782, 2571, 2592, 28170, 3001, 1012, 102, 101, 1999, 2023, 3259, 2057, 6235, 2122, 5461, 1998, 11628, 2037, 2836, 2408, 1037, 2846, 1997, 6254, 1010, 2147, 11066, 1010, 1998, 4094, 16820, 1012, 102, 101, 1031, 1016, 1033, 2574, 1010, 2172, 1997, 1996, 2951, 10573, 2058, 1996, 4274, 2097, 2022, 12359, 1999, 20950, 1010, 4352, 2005, 12138, 22910, 1998, 4180, 15058, 2094, 16972, 1012, 102, 101, 2057, 2031, 2328, 1037, 22910, 3194, 2170, 1061, 8873, 21928, 1010, 2029, 17736, 11058, 20950, 5491, 2429, 2000, 1060, 4226, 2854, 2030, 26726, 8988, 10861, 5134, 2008, 9125, 2119, 4130, 11423, 1998, 3653, 16467, 2015, 1012, 102, 101, 4406, 3025, 2147, 1010, 1061, 8873, 21928, 3594, 1037, 3117, 1050, 7011, 15058, 2094, 7781, 2944, 1012, 102, 101, 1999, 2023, 10467, 1010, 2057, 2556, 1996, 5090, 1998, 13792, 10318, 1061, 102], [1, 13971, 20952, 2050, 1031, 1014, 1033, 1010, 1998, 2060, 2512, 1011, 28283, 25300, 10074, 10713, 8285, 21022, 1006, 1050, 7011, 1007, 2241, 13792, 1006, 1060, 8873, 21928, 1031, 1015, 1033, 1010, 1061, 8873, 21928, 1031, 1016, 1033, 1010, 1998, 1060, 18886, 2063, 1031, 1017, 1033, 1007, 2031, 2042, 3818, 2004, 8114, 13792, 2005, 6364, 1037, 2312, 2193, 1997, 26726, 2229, 1999, 20950, 9199, 1012, 3, 13971, 20952, 2050, 1010, 2029, 2474, 28431, 1036, 1036, 9570, 2015, 28283, 25300, 10074, 10713, 8285, 21022, 1006, 1040, 7011, 1007, 1010, 2003, 6020, 2000, 1996, 2500, 1999, 3408, 1997, 6364, 2836, 1010, 2138, 2009, 16021, 14900, 1037, 5377, 2152, 2083, 18780, 1006, 2004, 3435, 2004, 2019, 20950, 11968, 8043, 1007, 2005, 1037, 3074, 1997, 2309, 1011, 4130, 26726, 2229, 1010, 2096, 1996, 2500, 1005, 2836, 2003, 7399, 2135, 26131, 2114, 1996, 2193, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 23, 88, 105, 142, 183, 226, 246, 271, 300, 322, 358, 384, 410, 441, 479, 498], [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ====================================================================================================
    """

    def batch_buffer(self, data, batch_size):
        # 最终返回的batch_buffer中example的长度在>=300
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)  # ex: example
            if(ex is None):  
                continue
            minibatch.append(ex)
            # 在abs模式下，函数中当前的batch_size指的是未来minibatch列表中所有example的tgt的token数量的一个上界，因此决定了minibatch列表中数据的长度；在ext模式下，函数中当前的batch_size指的是未来minibatch列表中所有example的src的token数量的一个上界
            # 在abs模式下，size_so_far是当前minibatch列表中以数据最大长度预估的一个tgt的token数，当然存在一个1e3的惩罚项；同理，在ext模式下，size_so_far是当前minibatch列表中以数据最大长度预估的一个src的token数
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            # 下面的if是为了保证size_so_far的token数永远小于等于batch_size表示的token数，从而决定了batch_buffer的数据长度
            if size_so_far == batch_size:  # size_so_far: 当前max_tgt_tokens*len(minibatch),  batch_size: self.batch_size * 300 例如：140*300；实际数值上batch_buffer >> 送进模型的batch长度
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        # 最后返回的batch size是个位数
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            # 类似batch_buffer中的comment，其中batch_size是命令行传进来的数值，通常设置为max_tgt_len相同的数值
            if size_so_far == batch_size:  # size_so_far: 当前max_tgt_tokens*len(minibatch), batch_size: 140 ; 140是参数中最大的tgt的token长度
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()  # 到底是torch.load的pt文件
        for buffer in self.batch_buffer(data, self.batch_size * 300):  # 300

            # 排序完成后的buffer实际会使得每个送入模型的每个batch中的数据比较均匀
            p_batch = sorted(buffer, key=lambda x: len(x[2]))  # 这句其实排了个寂寞序
            p_batch = sorted(p_batch, key=lambda x: len(x[1]))  # abs模式下按照example中tgt的长度排序

            p_batch = self.batch(p_batch, self.batch_size)  # 由于buffer经过了排序，所以先出来的batch的长度一定较长，这是受abs_batch_size_fn控制的


            p_batch = list(p_batch)
            if (self.shuffle):  # 单个minibatch内部再打乱顺序
                random.shuffle(p_batch)
            for b in p_batch:  # b的长度大概在6个左右
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        ## 理解代码参考的git issue：
        # https://github.com/nlpyang/PreSumm/issues/38
        # https://github.com/nlpyang/PreSumm/issues/8
        # https://github.com/nlpyang/PreSumm/issues/77
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:  # 如果在这个epoch中这些数据已经被迭代了，则快进，可能用在多进程中？
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return


class TextDataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.batch_size = batch_size
        self.device = device

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if (is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            p_batch = sorted(buffer, key=lambda x: len(x[2]))
            p_batch = sorted(p_batch, key=lambda x: len(x[1]))


            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
