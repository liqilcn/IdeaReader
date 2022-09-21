import gc
import os
import re
import json
import torch
import config
from tqdm import tqdm
from config import BertSumAbs_Preprocess_Args, BART_Preprocess_Args
from others.tokenization import BertTokenizer
from transformers import BartTokenizer
from prepro.utils import _get_word_ngrams
from prepro.dataset_tokenziers import dataset_tokenize   # 在config中定义好相应的预处理的数据集，dataset_tokenize会自动返回统一好格式的数据集的tokenize

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'  # 句子结束token
        self.cls_token = '[CLS]'  # CLS位token
        self.pad_token = '[PAD]'  # 空白占位符
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]  # 获得相应token的词表id
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]  # args.min_src_ntokens_per_sent：src中每个的句子的最小token个数，每个src往下分是句子，再往下分才是词

        _sent_labels = [0] * len(src)  # 每个句子的label，通过ROUGE贪心算法自动打标签，句子的label用于抽取式摘要生成，论文中有说明
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]  # args.min_src_ntokens_per_sent：src中每个的句子的最大token个数
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]  # 限制每个src中的最大sent个数
        sent_labels = sent_labels[:self.args.max_src_nsents] # 相应的句子label也随之改变

        if ((not is_test) and len(src) < self.args.min_src_nsents):  # src中的句子个数过少时返回None，此训练element被舍弃。
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)  # 在每个句子中插入[SEP]与[CLS]标志位

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]  # 前面join之后首位没有标志位，这步在整个src文本加上[CLS]与[SEP]标志位

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)  # 调用包将文本中的token转换为id
        # 下面这段实则是在给不同句子上的token分配不同的奇偶segment embedding，见论文
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]  # 获取每个[SEP] token在句子中的位置，那个-1用到下面，作为虚拟的句子开始0的前一个，后面是每个句子的开始token的前一位是前个句子的[SEP] token的位置
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))] # range从1开始，segs的第一个元素正好是从第一个[CLS]到该句子的[SEP]
        segments_ids = []  # 为每个句子的token赋予奇数或者偶数，用于索引后面输入Transformer的segment embedding。只需要奇偶就可以判别该句子与相邻前后两个句子不是同一个句子
        # 获取每个token的segment id
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]


        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt

'''
src_subtokens
# 每个句子用[CLS]与[SEP]包裹
['[CLS]', '[', '0', ']', 'a', 'continuous', 'nearest', 'neighbor', 'query', 'retrieve', '##s', 'the', 'nearest', 'neighbor', '(', 'n', '##n', ')', 'of', 'every', 'point', 'on', 'a', 'line', 'segment', '(', 'e', '##.', '##g', '.', '[SEP]', '[CLS]', '“', 'find', 'all', 'my', 'nearest', 'gas', 'stations', 'during', 'my', 'route', 'from', 'point', 's', 'to', 'point', 'e', '”', ')', '.', '[SEP]', '[CLS]', 'the', 'result', 'contains', 'a', 'set', 'of', 'point', ',', 'interval', 'tu', '##ples', ',', 'such', 'that', 'point', 'is', 'the', 'n', '##n', 'of', 'all', 'points', 'in', 'the', 'corresponding', 'interval', '.', '[SEP]', '[CLS]', 'existing', 'methods', 'for', 'continuous', 'nearest', 'neighbor', 'search', 'are', 'based', 'on', 'the', 'repetitive', 'application', 'of', 'simple', 'n', '##n', 'algorithms', ',', 'which', 'inc', '##urs', 'significant', 'overhead', '.', '[SEP]', '[CLS]', 'in', 'this', 'paper', 'we', 'propose', 'techniques', 'that', 'solve', 'the', 'problem', 'by', 'performing', 'a', 'single', 'query', 'for', 'the', 'whole', 'input', 'segment', '.', '[SEP]', '[CLS]', 'as', 'a', 'result', 'the', 'cost', ',', 'depending', 'on', 'the', 'query', 'and', 'data', '##set', 'characteristics', ',', 'may', 'drop', 'by', 'orders', 'of', 'magnitude', '.', '[SEP]', '[CLS]', 'in', 'addition', ',', 'we', 'propose', 'analytical', 'models', 'for', 'the', 'expected', 'size', 'of', 'the', 'output', ',', 'as', 'well', 'as', ',', 'the', 'cost', 'of', 'query', 'processing', ',', 'and', 'extend', 'out', 'techniques', 'to', 'several', 'variations', 'of', 'the', 'problem', '.', '[SEP]', '[CLS]', '[', '1', ']', 'this', 'paper', 'addresses', 'the', 'problem', 'of', 'monitoring', 'the', 'k', 'nearest', 'neighbors', 'to', 'a', 'dynamic', '##ally', 'changing', 'path', 'in', 'road', 'networks', '.', '[SEP]', '[CLS]', 'given', 'a', 'destination', 'where', 'a', 'user', 'is', 'going', 'to', ',', 'this', 'new', 'query', 'returns', 'the', 'k', 'n', '##n', 'with', 'respect', 'to', 'the', 'shortest', 'path', 'connecting', 'the', 'destination', 'and', 'the', 'user', "'", '##s', 'current', 'location', ',', 'and', 'thus', 'provides', 'a', 'list', 'of', 'nearest', 'candidates', 'for', 'reference', 'by', 'considering', 'the', 'whole', 'coming', 'journey', '.', '[SEP]', '[CLS]', 'we', 'name', 'this', 'query', 'the', 'k', 'path', 'nearest', 'neighbor', 'query', '(', 'k', 'p', '##nn', ')', '.', '[SEP]', '[CLS]', 'as', 'the', 'user', 'is', 'moving', 'and', 'may', 'not', 'always', 'follow', 'the', 'shortest', 'path', ',', 'the', 'query', 'path', 'keeps', 'changing', '.', '[SEP]', '[CLS]', 'the', 'challenge', 'of', 'monitoring', 'the', 'k', 'p', '##nn', 'for', 'an', 'ar', '##bit', '##rar', '##ily', 'moving', 'user', 'is', 'to', 'dynamic', '##ally', 'determine', 'the', 'update', 'locations', 'and', 'then', 'ref', '##resh', 'the', 'k', 'p', '##nn', 'efficiently', '.', '[SEP]', '[CLS]', 'we', 'propose', 'a', 'three', '##pha', '##se', 'best', '##fi', '##rst', 'network', 'expansion', '(', 'bn', '##e', ')', 'algorithm', 'for', 'monitoring', 'the', 'k', 'p', '##nn', 'and', 'the', 'corresponding', 'shortest', 'path', '.', '[SEP]', '[CLS]', 'in', 'the', 'searching', 'phase', ',', 'the', 'bn', '##e', 'finds', 'the', 'shortest', 'path', 'to', 'the', 'destination', ',', 'during', 'which', 'a', 'candidate', 'set', 'that', 'guarantees', 'to', 'include', 'the', 'k', 'p', '##nn', 'is', 'generated', 'at', 'the', 'same', 'time', '.', '[SEP]', '[CLS]', 'then', 'in', 'the', 'verification', 'phase', ',', 'a', 'he', '##uri', '##stic', 'algorithm', 'runs', 'for', 'examining', 'candidates', "'", 'exact', 'distances', 'to', 'the', 'query', 'path', ',', 'and', 'it', 'achieve', '##s', 'significant', 'reduction', 'in', 'the', 'number', 'of', 'visited', 'nodes', '.', '[SEP]', '[CLS]', 'the', 'monitoring', 'phase', 'deals', 'with', 'computing', 'update', 'locations', 'as', 'well', 'as', 'refreshing', 'the', 'k', 'p', '##nn', 'in', 'different', 'user', 'movements', '.', '[SEP]', '[CLS]', 'since', 'determining', 'the', 'network', 'distance', 'is', 'a', 'costly', 'process', ',', 'an', 'expansion', 'tree', 'and', 'the', 'candidate', 'set', 'are', 'carefully', 'maintained', 'by', 'the', 'bn', '##e', 'algorithm', ',', 'which', 'can', 'provide', 'efficient', 'update', 'on', 'the', 'shortest', 'path', 'and', 'the', 'k', 'p', '##nn', 'results', '.', '[SEP]', '[CLS]', 'finally', ',', 'we', 'conduct', 'extensive', 'experiments', 'on', 'real', 'road', 'networks', 'and', 'show', 'that', 'our', 'methods', 'achieve', 'satisfactory', 'performance', '.', '[SEP]', '[CLS]', '[', '2', ']', 'tr', '##aj', '##ect', '##ories', 'representing', 'the', 'motion', 'of', 'moving', 'objects', 'are', 'typically', 'obtained', 'via', 'location', 'sampling', ',', 'e', '##.', '##g', '.', '[SEP]', '[CLS]', 'using', 'gps', 'or', 'roadside', 'sensors', ',', 'at', 'discrete', 'time', '##ins', '##tan', '##ts', '.', '[SEP]', '[CLS]', 'in', '##bet', '##wee', '##n', 'consecutive', 'samples', ',', 'nothing', 'is', 'known', 'about', 'the', 'whereabouts', 'of', 'a', 'given', 'moving', 'object', '.', '[SEP]', '[CLS]', 'various', 'models', 'have', 'been', 'proposed', '(', 'e', '##.', '##g', '.', '[SEP]', '[CLS]', 'shear', '##ed', 'cylinders', ';', 'space', '##time', 'prism', '##s', ')', 'to', 'represent', 'the', 'uncertainty', 'of', 'the', 'moving', 'objects', 'both', 'in', 'un', '##con', '##stra', '##ined', 'eu', '##cl', '##idia', '##n', 'space', ',', 'as', 'well', 'as', 'road', 'networks', '.', '[SEP]', '[CLS]', 'in', 'this', 'paper', ',', 'we', 'focus', 'on', 'representing', 'the', 'uncertainty', 'of', 'the', 'objects', 'moving', 'along', 'road', 'networks', 'as', 'timed', '##ep', '##end', '##ent', 'probability', 'distribution', 'functions', ',', 'assuming', 'availability', 'of', 'a', 'maximal', 'speed', 'on', 'each', 'road', 'segment', '.', '[SEP]', '[CLS]', 'for', 'these', 'settings', ',', 'we', 'introduce', 'a', 'novel', 'index', '##ing', 'mechanism', 'ut', '##h', '(', 'uncertain', 'tr', '##aj', '##ect', '##ories', 'hierarchy', ')', ',', 'based', 'upon', 'which', 'efficient', 'algorithms', 'for', 'processing', 'spat', '##iot', '##em', '##por', '##al', 'range', 'que', '##ries', 'are', 'proposed', '.', '[SEP]', '[CLS]', 'we', 'also', 'present', 'experimental', 'results', 'that', 'demonstrate', 'the', 'benefits', 'of', 'our', 'proposed', 'method', '##ologies', '.', '[SEP]']
tgt_subtoken
# tgt长文本的开头和结尾分别为[unused0]与[unused1]，中间的句子用[unused2]分隔
['[unused0]', 'many', 'works', 'focus', 'on', 'kn', '##n', 'que', '##ries', 'from', 'mobile', 'devices', 'on', 'static', 'objects', 'with', 'indices', 'built', 'on', 'static', 'points', 'and', 'road', 'network', 'data', '.', '[unused2]', 'in', ',', 'methods', 'are', 'proposed', 'to', 'reduce', 'the', 'cost', 'of', 'each', 'query', 'operation', 'by', 'using', 'information', 'of', 'previous', 'que', '##ries', 'and', 'pre', '-', 'fetch', '##ed', 'results', 'that', 'are', 'stored', 'in', 'an', 'r', '-', 'tree', '.', '[unused2]', 'in', '[', '0', ']', ',', 'an', 'algorithm', 'is', 'proposed', 'to', 'find', 'the', 'kn', '##ns', 'for', 'all', 'positions', 'by', 'searching', 'an', 'r', '-', 'tree', 'only', 'once', '.', '[unused2]', 'in', '[', '1', ']', ',', 'best', '-', 'first', 'network', 'expansion', '(', 'bn', '##e', ')', 'algorithm', 'is', 'proposed', 'for', 'monitoring', 'k', '-', 'path', 'nearest', 'neighbor', '(', 'k', '##p', '##nn', ')', 'que', '##ries', ',', 'where', 'an', 'expansion', 'tree', 'and', 'a', 'candidate', 'set', 'are', 'utilized', 'for', 'efficient', 'k', '##p', '##nn', 'updates', 'and', 'results', '.', '[unused2]', 'in', '[', '2', ']', ',', 'uncertain', 'tr', '##aj', '##ect', '##ories', 'hierarchy', '(', 'ut', '##h', ')', 'is', 'proposed', 'to', 'process', 'spat', '##io', '-', 'temporal', 'range', 'que', '##ries', 'for', 'uncertain', 'tr', '##aj', '##ect', '##ories', 'on', 'road', 'networks', '.', '[unused2]', 'in', 'ut', '##h', ',', 'for', 'each', 'edge', ',', 'time', 'periods', 'are', 'stored', 'in', 'an', 'r', '-', 'tree', 'within', 'which', 'objects', 'are', 'moving', '.', '[unused1]']
tgt_txt
# 用<q>分隔句子
many works focus on knn queries from mobile devices on static objects with indices built on static points and road network data .<q>in , methods are proposed to reduce the cost of each query operation by using information of previous queries and pre-fetched results that are stored in an r-tree .<q>in [ 0 ] , an algorithm is proposed to find the knns for all positions by searching an r-tree only once .<q>in [ 1 ] , best-first network expansion ( bne ) algorithm is proposed for monitoring k-path nearest neighbor ( kpnn ) queries , where an expansion tree and a candidate set are utilized for efficient kpnn updates and results .<q>in [ 2 ] , uncertain trajectories hierarchy ( uth ) is proposed to process spatio-temporal range queries for uncertain trajectories on road networks .<q>in uth , for each edge , time periods are stored in an r-tree within which objects are moving .
src_txt
句子文本的列表
['[ 0 ] a continuous nearest neighbor query retrieves the nearest neighbor ( nn ) of every point on a line segment ( e.g .', '“ find all my nearest gas stations during my route from point s to point e ” ) .', 'the result contains a set of point , interval tuples , such that point is the nn of all points in the corresponding interval .', 'existing methods for continuous nearest neighbor search are based on the repetitive application of simple nn algorithms , which incurs significant overhead .', 'in this paper we propose techniques that solve the problem by performing a single query for the whole input segment .', 'as a result the cost , depending on the query and dataset characteristics , may drop by orders of magnitude .', 'in addition , we propose analytical models for the expected size of the output , as well as , the cost of query processing , and extend out techniques to several variations of the problem .', '[ 1 ] this paper addresses the problem of monitoring the k nearest neighbors to a dynamically changing path in road networks .', "given a destination where a user is going to , this new query returns the k nn with respect to the shortest path connecting the destination and the user 's current location , and thus provides a list of nearest candidates for reference by considering the whole coming journey .", 'we name this query the k path nearest neighbor query ( k pnn ) .', 'as the user is moving and may not always follow the shortest path , the query path keeps changing .', 'the challenge of monitoring the k pnn for an arbitrarily moving user is to dynamically determine the update locations and then refresh the k pnn efficiently .', 'we propose a threephase bestfirst network expansion ( bne ) algorithm for monitoring the k pnn and the corresponding shortest path .', 'in the searching phase , the bne finds the shortest path to the destination , during which a candidate set that guarantees to include the k pnn is generated at the same time .', "then in the verification phase , a heuristic algorithm runs for examining candidates ' exact distances to the query path , and it achieves significant reduction in the number of visited nodes .", 'the monitoring phase deals with computing update locations as well as refreshing the k pnn in different user movements .', 'since determining the network distance is a costly process , an expansion tree and the candidate set are carefully maintained by the bne algorithm , which can provide efficient update on the shortest path and the k pnn results .', 'finally , we conduct extensive experiments on real road networks and show that our methods achieve satisfactory performance .', '[ 2 ] trajectories representing the motion of moving objects are typically obtained via location sampling , e.g .', 'using gps or roadside sensors , at discrete timeinstants .', 'inbetween consecutive samples , nothing is known about the whereabouts of a given moving object .', 'various models have been proposed ( e.g .', 'sheared cylinders ; spacetime prisms ) to represent the uncertainty of the moving objects both in unconstrained euclidian space , as well as road networks .', 'in this paper , we focus on representing the uncertainty of the objects moving along road networks as timedependent probability distribution functions , assuming availability of a maximal speed on each road segment .', 'for these settings , we introduce a novel indexing mechanism uth ( uncertain trajectories hierarchy ) , based upon which efficient algorithms for processing spatiotemporal range queries are proposed .', 'we also present experimental results that demonstrate the benefits of our proposed methodologies .']
其他参数：
src_subtoken_idxs：src的token对应的BERT词表的ID
sent_labels：每个sent的label，用于抽取式总结
tgt_subtoken_idxs：tgt的token对应的BERT词表的ID
segments_ids，src的token对应的segment id见论文
cls_ids，[CLS]token对应的位置
'''
def format_to_bertsumabs():
    # 将数据集中train，valid，test中的数据按照config.SHARD分成多个切片并格式化为BERTSUM的格式存储在.pt文件中
    # 引入config中的超参数
    args = BertSumAbs_Preprocess_Args()
    bert = BertData(args)
    dataset = dataset_tokenize()  # dataset_tokenize返回的格式仍然是多文档总结的格式，需要先将多个src拼接成一个src
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)

    for ds_type in dataset:
        all_data_eles = dataset[ds_type]
        is_test = ds_type == 'test'
        data_shard = []
        shard_ct = 0
        for data in all_data_eles:
            source  = []
            for doc in data['multi_doc']:  # 将多个文档合并成一个文档，BERTSUM是单文档总结
                source += doc
            tgt = data['abs']
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
            b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=True,
                                 is_test=is_test)
            if (b_data is None):
                continue
            src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,  # 这里是预处理结束后pt文件中存储的最终数据格式
                           "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,  # 里面元素具体来自于bert.preprocess，有详细例子
                           'src_txt': src_txt, "tgt_txt": tgt_txt}
            data_shard.append(b_data_dict)
            if len(data_shard) == args.shard_size:
                file_name = f'{ds_type}.{shard_ct}.pt'
                save_file = os.path.join(args.save_path, file_name)
                print(f'Saving to {save_file}')
                torch.save(data_shard, save_file)
                data_shard = []
                shard_ct += 1
        if len(data_shard) > 0:
            file_name = f'{ds_type}.{shard_ct}.pt'
            save_file = os.path.join(args.save_path, file_name)
            print(f'Saving to {save_file}')
            torch.save(data_shard, save_file)

    gc.collect()

def format_to_bart():
    args = BART_Preprocess_Args()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large" if args.large else "facebook/bart-base")
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    for ds_type in ['train', 'validate', 'test']:
        print(f'preprocess {ds_type}.json...')
        with open(os.path.join(config.raw_dataset_path, f'{ds_type}.json'), 'r') as fp:
            jsonlines = fp.readlines()
            datas = []
            for line in tqdm(jsonlines):
                json_line = json.loads(line)
                source = ' '.join(json_line['multi_doc'])
                tgt = json_line['abs']
                token_len_src = len(tokenizer.tokenize(source))
                token_len_tgt = len(tokenizer.tokenize(tgt))
                if token_len_src >= args.min_src_ntokens and token_len_tgt >= args.min_tgt_ntokens:
                    datas.append({'src': source, 'ref': tgt})
            save_file = os.path.join(args.save_path, f'{ds_type}.pt')
            torch.save(datas, save_file)

