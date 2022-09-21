import os
# data preprocessing多个数据集处理成多个模型所需要的不同数据格式
# 会有多个不同格式的数据集，会有多个用于比较的模型，或许需要
preprocess_dataset = 'delve'  # 可选delve，s2orc
# target_model = 'bertsumabs'  # 可选bertsumabs，最终会将相应的.pt文件写入该模型对应的数据目录中，torch_data下对应各个模型所需的数据集
target_model = 'bart'

##########################################数据预处理##########################################
raw_dataset_path = f'../raw_data/{preprocess_dataset}'
lower = True  # 是否将数据集全部小写化；小写化处理在每个数据集的tokenize进行处理比较稳妥，而delve数据集均为小写，故不需要小写化


# bertsum模型所需数据处理专有超参，在具体的format_to_bert_sum中调用
class BertSumAbs_Preprocess_Args():
    def __init__(self):
        self.shard_size = 2000  # 对数据进行预处理时的每个shard的样本长度
        self.min_src_nsents = 3
        self.max_src_nsents = 100
        self.min_src_ntokens_per_sent = 5
        self.max_src_ntokens_per_sent = 200
        self.min_tgt_ntokens = 5
        self.max_tgt_ntokens = 500
        self.save_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'

##########################################模型训练，验证，测试##########################################

class BertSumAbs_Train_Args():
    def __init__(self):
        self.encoder = 'bert'  #
        self.mode = 'validate'  # choices=['train', 'validate', 'test']
        self.torch_data_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'  # 预处理完成后pt文件的路径
        self.model_path = f'../models/{target_model}_model/{preprocess_dataset}'  # cp路径，分模型，分数据集
        self.result_path = f'../results/{target_model}_result/{preprocess_dataset}'  # 输出结果，分模型，分数据集
        self.temp_dir = '../temp'

        self.batch_size = 140  # default=140
        self.test_batch_size = 200 # default=200
        self.max_pos = 512 # default=512
        self.use_interval = True  # choices=[True, False], default=True
        self.large = False  # choices=[True, False], default=False
        self.load_from_extractive = ''

        self.sep_optim = True  # default=False
        self.lr_bert = 0.002  # default=2e-3  # bert微调的学习率也很小，这里是0.002，实际在1e-5数量级
        self.lr_dec = 0.2  # default=2e-3
        self.use_bert_emb = True  # default=False

        self.share_emb = False  # default=False
        self.finetune_bert = True  # default=True
        self.dec_dropout = 0.2  # default=0.2
        self.dec_layers = 6  # default=6
        self.dec_hidden_size = 768  # default=768
        self.dec_heads = 8  # default=8
        self.dec_ff_size = 2048  # default=2048
        self.enc_hidden_size = 512  # default=512
        self.enc_ff_size = 512  # default=512
        self.enc_dropout = 0.2  # default=0.2
        self.enc_layers = 6  # default=6

        self.label_smoothing = 0.1  # default=0.1
        self.generator_shard_size = 32  # default=32
        self.alpha = 0.6  # default=0.6
        self.beam_size = 5  # default=5
        self.min_length = 50  # default=15
        self.max_length = 100  # default=150
        self.max_tgt_len = 140  # default=140

        self.param_init = 0  # default=0
        self.param_init_glorot = True  # default=True
        self.optim = 'adam'  # default='adam'
        self.lr = 1  # default=1
        self.beta1 = 0.9  # default=0.9
        self.beta2 = 0.999  # default=0.999
        self.warmup_steps = 8000  # default=8000
        self.warmup_steps_bert = 20000  # default=8000
        self.warmup_steps_dec = 10000  # default=8000
        self.max_grad_norm = 0  # default=0

        self.save_checkpoint_steps = 5000  # default=5
        self.accum_count = 5  # default=1
        self.report_every = 50  # default=1
        self.train_steps = 600000  # default=1000
        self.recall_eval = False  # default=False

        self.visible_gpus = '1'  # default='-1'
        self.gpu_ranks = '0'  # default='0'
        if not os.path.exists('../logs'): os.makedirs('../logs')
        self.log_file = f'../logs/{target_model}_{preprocess_dataset}.log'  # default='../logs/delve.log'
        self.seed = 666  # default=666

        self.test_all = True  # default=False
        self.test_from = ''  # default=''
        self.test_start_from = 100000  # default=-1

        self.train_from = ''  # default=''
        self.report_rouge = True  # default=True
        self.block_trigram = True  # default=True


class BART_Preprocess_Args():
    def __init__(self):
        self.save_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'
        self.large = False
        self.min_src_ntokens = 100
        self.min_tgt_ntokens = 50

class BART_Train_Args():
    def __init__(self):
        self.mode = 'validate'  # choices=['train', 'validate']  # valid的时候就已经完成了test
        self.torch_data_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'  # 预处理完成后pt文件的路径
        self.model_path = f'../models/{target_model}_model/{preprocess_dataset}'  # cp路径，分模型，分数据集
        self.result_path = f'../results/{target_model}_result/{preprocess_dataset}'  # 输出结果，分模型，分数据集
        if not os.path.exists(self.result_path): os.makedirs(self.result_path)
        self.temp_dir = '../temp'

        self.batch_size = 10
        self.test_batch_size = 25  # batch过大会导致内存溢出，batch的长度就是batch中的item的数量
        self.large = False  # choices=[True, False], default=False

        # 文本生成时用的参数
        self.beam_size = 5
        self.min_length = 50
        self.max_length = 100

        # 训练用的参数
        self.src_max_length = 512
        self.ref_max_length = 140

        self.lr = 0.002  # 其他finetune BART的论文中使用的学习率大概1e-5，因此需要调小这个值防止梯度爆炸以及loss突然增加使得训练停止，参考：https://arxiv.org/pdf/2003.02245.pdf
        # finetue大模型如果学习率太大会在训练过程发生梯度爆炸
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.warmup_steps = 8000
        self.max_grad_norm = 0

        self.save_checkpoint_steps = 5000
        self.accum_count = 5
        self.report_every = 20
        self.train_steps = 600000
        self.recall_eval = False

        self.visible_gpus = '0'
        self.gpu_ranks = '0'
        if not os.path.exists('../logs'): os.makedirs('../logs')
        self.log_file = f'../logs/{target_model}_{preprocess_dataset}.log'
        self.seed = 666

        self.test_all = True
        self.test_from = ''
        self.test_start_from = 25000

        self.train_from = f'../models/{target_model}_model/{preprocess_dataset}/model_step_40000.pt'
        self.report_rouge = True
        self.block_trigram = True