import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.optimizers import Optimizer

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
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
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
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
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
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
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)  # 对变换后的隐向量做softmax
    generator = nn.Sequential(  
        nn.Linear(dec_hidden_size, vocab_size), # 先对decoder输出的隐向量做线性变换，将768维的隐变量变换为词表长度，然后将词表长度的向量做log(softmax)，得到一个词表长度的概率分布（取log后是负无穷到0的分布）
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)  # from_pretrained用pretrain好的模型初始化当前的Bertsum
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()  # 启用测试模式，关闭Dropout并使用训练学习到的BN的均值和方差
            with torch.no_grad():  # 由于关闭了finetune模式，所以不需要更新梯度，with torch.no_grad()就是停止autograd模块的工作，以起到加速和节省显存的作用，具体行为就是停止gradient计算
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:  # 用抽取式模型finetune好bert之后再来初始化生成式的模型
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config) # 这里直接用BertModel(bert_config)初始化model的话，并未加载预训练的参数，而是从头训练BERT的transformer

        if(args.max_pos>512): # 前512个token用bert自带的位置编码，超过512的token是bert的最后一位token的位置编码
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)  # 返回一个查找表，每个词的序号对应一个特定维数的词向量
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data  # 是指取出所有的词向量my_pos_embeddings.weight.data，可以直接索引，每个索引代表相应的词向量
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)  # 用bert的最后一个位置编码填充token超过512的token的位置编码
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)  # padding_idx即pad token所对应的id，默认为0
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)  # 复制词向量的参数

        self.decoder = TransformerDecoder(  # 不去看源代码了，源码乱七八糟，无文档，用的时候直接用pytorch自带的transformer的解码器https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight  # 用nn.embedding自己定义时初始化的参数对self.generator中的nn.Linear()进行初始化


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)  # 从checkpoint加载模型参数
        else:
            for module in self.decoder.modules():  # 若不加载checkpoint，则对decoder各个模块的参数进行初始化
                if isinstance(module, (nn.Linear, nn.Embedding)):  # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)  # attention_mask 可选。各元素的值为 0 或 1 ，避免在 padding 的 token 上计算 attention（1不进行masked，0则masked）。形状为(batch_size, sequence_length)。
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        # 仍以“我爱你”到“I love you”为例，在训练过程中，encoder输⼊的是待翻译句⼦“我爱你”，
        # 得到top_vec；encoder输⼊的是“S I love you”（S位起始符）；最终计算loss的label是“I love you E”。
        # “S I love you E”不只用来计算loss，还用来加上mask之后预测下一个词的decoder的输入
        # 例如：tgt为：S I love you E，tgt[:, :-1]为S I love you，真实的label为I love you E
        # tgt[:, :-1]是因为tgt的最后一位用于decoder输入的时候总是没有意义的，
        # 在decoder输入的tgt的倒数第二位实际为真实文本的最后一位，
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
