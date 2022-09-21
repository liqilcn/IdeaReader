import os
import torch
from tensorboardX import SummaryWriter
from transformers import BartTokenizer

import distributed
from models.reporter_bart import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str

def batch_detail(batch, step):
    src_len_list = []
    ref_len_list = []
    srcs, refs = batch
    for item in srcs:
        src_len_list.append(len(item.split(' ')))
    for item in refs:
        ref_len_list.append(len(item.split(' ')))
    # print(f'step {step}; rank: {get_rank()}; src : {sorted(src_len_list, reverse=True)}; ref: {sorted(ref_len_list, reverse=True)};')
    logger.info(f'step {step}; src : {sorted(src_len_list, reverse=True)}; ref: {sorted(ref_len_list, reverse=True)};')

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count  # 梯度积累，相当于在低显存下增加batch，用于单机多卡训练收集多个batch，然后到达这个数之后，分发给多个卡进行梯度下降
    n_gpu = args.world_size  # 使用的GPU数量

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])  # 单进程或者多进程下相应的gpu_id
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")  # tensorboard画板

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)


    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)  # 最终的dataloader的迭代训练

    # print(tr)
    if (model):  # 总共的参数个数
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, optim,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.args = args
        self.device = "cpu" if args.visible_gpus == '-1' else "cuda"
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large" if args.large else "facebook/bart-base")
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager  # report_manager对象

        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optim._step + 1  # 用于控制训练的步数

        true_batchs = []
        accum = 0
        train_iter = train_iter_fct()

        # reporter, 不影响训练的逻辑
        report_stats = Statistics()  # Statistics对象，一个batch更新一次，训练中当一个report_stats被打印到标准输出流之后，report_stats随即别重置返回一个新的Statistics对象
        self._start_report_manager(start_time=report_stats.start_time)  # 将当前的时间作为report_manager的起始时间

        while step <= train_steps:

            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank): # 不同的GPU会分到不同的batch
                    true_batchs.append(batch) # 当不在梯度累加模式下，true_batchs中只有一个batch
                    accum += 1  # 记录当前true_batchs中累积的batch个数，先累计到self.grad_accum_count个，然后再交给多个进程进行参数更新
                    if accum == self.grad_accum_count:  # 收集完self.grad_accum_count数量的batch之后，将多个batch分给多个卡，使用self._gradient_accumulation统一进行梯度下降
                        self._gradient_accumulation(  # 完成反向传播，梯度累加，以及单，多GPU模式下的参数更新
                            true_batchs, report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)
                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src, ref = batch
                dct = self.tokenizer.batch_encode_plus(src, max_length=self.args.src_max_length, return_tensors="pt",
                                                       truncation=True, padding=True)
                ans = self.tokenizer.batch_encode_plus(ref, max_length=self.args.ref_max_length, return_tensors="pt",
                                                       truncation=True, padding=True)
                #
                pad_token_id = self.tokenizer.pad_token_id
                y = ans['input_ids']
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone()
                lm_labels[y[:, 1:] == pad_token_id] = -100

                output = self.model(
                    input_ids=dct['input_ids'].to(self.device),
                    attention_mask=dct['attention_mask'].to(self.device),
                    decoder_input_ids=y_ids.to(self.device),
                    labels=lm_labels.to(self.device),
                )
                loss = output[0]

                batch_stats = Statistics(loss.item(), 1)
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def _gradient_accumulation(self, true_batchs, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()  # 即将进行反向传播，将梯度清零

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()
            # batch_detail(batch, step)
            src, ref = batch
            dct = self.tokenizer.batch_encode_plus(src, max_length=self.args.src_max_length, return_tensors="pt",
                                                   truncation=True, padding=True)
            ans = self.tokenizer.batch_encode_plus(ref, max_length=self.args.ref_max_length, return_tensors="pt",
                                                   truncation=True, padding=True)
            pad_token_id = self.tokenizer.pad_token_id
            y = ans['input_ids']
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100

            output = self.model(
                input_ids=dct['input_ids'].to(self.device),
                attention_mask=dct['attention_mask'].to(self.device),
                decoder_input_ids=y_ids.to(self.device),
                labels=lm_labels.to(self.device),
            )
            loss = output[0]
            loss.backward()
            report_stats.update(Statistics(loss.item(), 1))

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:  # 不进行梯度积累，每个GPU处理一个batch
                # Multi GPU gradient gather
                if self.n_gpu > 1:  # 若并行，则将所有梯度（下面的grads）收集好传给主GPU进行参数更新
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(  # 分布式参数更新（reduce），即将所有GPU上batch的梯度再次在主GPU进行累积，等待最终的梯度下降
                        grads, float(1))
                self.optim.step()


        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:  # self.grad_accum_count > 1的情况，单个GPU上先使得梯度在多个batch下累加，然后统一由主GPU更新梯度
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {  # checkpoint的pt的具体格式
            'model': model_state_dict,  # 模型的参数
            # 'generator': generator_state_dict,
            'opt': self.args,  # 训练用的超参数
            'optim': self.optim,  # 优化器的参数
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        # 将当前时间设置为report manager的起始时间
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None): # 验证的时候learning_rate设置为0
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
