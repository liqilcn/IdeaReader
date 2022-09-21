import os

preprocess_dataset = 'delve'  # optional parameters: 'delve', 's2orc'
target_model = 'bertsumabs'  # optional parameters: 'bertsumabs', 'bart'

raw_dataset_path = f'../raw_data/{preprocess_dataset}'
lower = True

# args for bertsumabs data preprocessing
class BertSumAbs_Preprocess_Args():
    def __init__(self):
        self.shard_size = 2000
        self.min_src_nsents = 3
        self.max_src_nsents = 100
        self.min_src_ntokens_per_sent = 5
        self.max_src_ntokens_per_sent = 200
        self.min_tgt_ntokens = 5
        self.max_tgt_ntokens = 500
        self.save_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'

# args for bertsumabs training
class BertSumAbs_Train_Args():
    def __init__(self):
        self.encoder = 'bert'
        self.mode = 'train'  # choices=['train', 'validate', 'test']
        self.torch_data_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'
        self.model_path = f'../models/{target_model}_model/{preprocess_dataset}'
        self.result_path = f'../results/{target_model}_result/{preprocess_dataset}'
        self.temp_dir = '../temp'

        self.batch_size = 140
        self.test_batch_size = 200
        self.max_pos = 512
        self.use_interval = True
        self.large = False
        self.load_from_extractive = ''

        self.sep_optim = True
        self.lr_bert = 0.002
        self.lr_dec = 0.2
        self.use_bert_emb = True

        self.share_emb = False
        self.finetune_bert = True
        self.dec_dropout = 0.2
        self.dec_layers = 6
        self.dec_hidden_size = 768
        self.dec_heads = 8
        self.dec_ff_size = 2048
        self.enc_hidden_size = 512
        self.enc_ff_size = 512
        self.enc_dropout = 0.2
        self.enc_layers = 6

        self.label_smoothing = 0.1
        self.generator_shard_size = 32
        self.alpha = 0.6
        self.beam_size = 5
        self.min_length = 50
        self.max_length = 100
        self.max_tgt_len = 140

        self.param_init = 0
        self.param_init_glorot = True
        self.optim = 'adam'
        self.lr = 1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.warmup_steps = 8000
        self.warmup_steps_bert = 20000
        self.warmup_steps_dec = 10000
        self.max_grad_norm = 0

        self.save_checkpoint_steps = 5000
        self.accum_count = 5
        self.report_every = 50
        self.train_steps = 600000
        self.recall_eval = False

        self.visible_gpus = '0,1,2'
        self.gpu_ranks = '0'
        if not os.path.exists('../logs'): os.makedirs('../logs')
        self.log_file = f'../logs/{target_model}_{preprocess_dataset}.log'
        self.seed = 666

        self.test_all = False
        self.test_from = ''
        self.test_start_from = -1

        self.train_from = ''
        self.report_rouge = True
        self.block_trigram = True

# args for bertsumabs verification
class BertSumAbs_Train_Args():
    def __init__(self):
        self.encoder = 'bert'
        self.mode = 'validate'  # choices=['train', 'validate']
        self.torch_data_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'
        self.model_path = f'../models/{target_model}_model/{preprocess_dataset}'
        self.result_path = f'../results/{target_model}_result/{preprocess_dataset}'
        self.temp_dir = '../temp'

        self.batch_size = 140
        self.test_batch_size = 200
        self.max_pos = 512
        self.use_interval = True
        self.large = False
        self.load_from_extractive = ''

        self.sep_optim = True
        self.lr_bert = 0.002
        self.lr_dec = 0.2
        self.use_bert_emb = True

        self.share_emb = False
        self.finetune_bert = True
        self.dec_dropout = 0.2
        self.dec_layers = 6
        self.dec_hidden_size = 768
        self.dec_heads = 8
        self.dec_ff_size = 2048
        self.enc_hidden_size = 512
        self.enc_ff_size = 512
        self.enc_dropout = 0.2
        self.enc_layers = 6

        self.label_smoothing = 0.1
        self.generator_shard_size = 32
        self.alpha = 0.6
        self.beam_size = 5
        self.min_length = 50
        self.max_length = 100
        self.max_tgt_len = 140

        self.param_init = 0
        self.param_init_glorot = True
        self.optim = 'adam'
        self.lr = 1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.warmup_steps = 8000
        self.warmup_steps_bert = 20000
        self.warmup_steps_dec = 10000
        self.max_grad_norm = 0

        self.save_checkpoint_steps = 5000
        self.accum_count = 5
        self.report_every = 50
        self.train_steps = 600000
        self.recall_eval = False

        self.visible_gpus = '1'
        self.gpu_ranks = '0'
        if not os.path.exists('../logs'): os.makedirs('../logs')
        self.log_file = f'../logs/{target_model}_{preprocess_dataset}.log'
        self.seed = 666

        self.test_all = True
        self.test_from = ''
        self.test_start_from = 100000

        self.train_from = ''
        self.report_rouge = True
        self.block_trigram = True

# args for bart data preprocessing
class BART_Preprocess_Args():
    def __init__(self):
        self.save_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'
        self.large = False
        self.min_src_ntokens = 100
        self.min_tgt_ntokens = 50

# args for bart train
class BART_Train_Args():
    def __init__(self):
        self.mode = 'train'
        self.torch_data_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'
        self.model_path = f'../models/{target_model}_model/{preprocess_dataset}'
        self.result_path = f'../results/{target_model}_result/{preprocess_dataset}'
        if not os.path.exists(self.result_path): os.makedirs(self.result_path)
        self.temp_dir = '../temp'

        self.batch_size = 10
        self.test_batch_size = 25
        self.large = False

        self.beam_size = 5
        self.min_length = 50
        self.max_length = 100

        self.src_max_length = 512
        self.ref_max_length = 140

        self.lr = 0.002
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

        self.train_from = ''
        self.report_rouge = True
        self.block_trigram = True

# args for bart verification
class BART_Train_Args():
    def __init__(self):
        self.mode = 'validate'
        self.torch_data_path = f'../torch_data/{target_model}_data/{preprocess_dataset}'
        self.model_path = f'../models/{target_model}_model/{preprocess_dataset}'
        self.result_path = f'../results/{target_model}_result/{preprocess_dataset}'
        if not os.path.exists(self.result_path): os.makedirs(self.result_path)
        self.temp_dir = '../temp'

        self.batch_size = 10
        self.test_batch_size = 25
        self.large = False

        self.beam_size = 5
        self.min_length = 50
        self.max_length = 100

        self.src_max_length = 512
        self.ref_max_length = 140

        self.lr = 0.002
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

        self.train_from = ''