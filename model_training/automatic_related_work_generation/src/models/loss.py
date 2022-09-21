# 有关文本生成loss计算的理解（已经无问题）
# loss计算是针对单个batch内数据计算的
# 在训练过程中生成文本长度总是和ground truth的长度相同
# 归根到底为单个词的多分类问题，类的数量为词表的长度
# 每生成一个词，softmax预测的过程就会有一个分布，长度为词表长度，对应的ground truth中相同位置上的词也有一个分布，长度为词表长度，该词对应位置的值为1，其余位置为0
# 在loss计算过程中，会将batch中多个item的预测token以及ground truth token合并，
# 具体对于ground truth来说，合并后的二维矩阵行代代表整个batch中所有item的ground truth中有效token，行数为相应有效token的数量，
# 单个行为对应token的概率分布，单个行的和为1，ground truth的行只有对应token所在的词表位置的元素为1，其余均为0
# 合并后的预测结果矩阵也和ground truth的尺寸相同
# 使用F.kl_div(output, model_prob, reduction='sum')之后，预测结果矩阵与ground truth矩阵每个行之间都会有一个散度值
# 上述函数将batch中ground truth的有效token个数的散度值求和，即将reduction设为'sum'
# 最后将和再除以一个batch中ground truth的有效token个数，则得到最终的loss
# 所以文本生成的loss是平均到每个token的数值
"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reporter import Statistics


def abs_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0):
    compute = NMTLossCompute(
        generator, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute



class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, pad_id):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = pad_id



    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, output)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output,
                              shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = Statistics()
        shard_state = self._make_shard_state(batch, output)  # trainer传过来的decoder的output
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()  # 除normalization会使得整体loss对参数的导数乘1/normalization, 因为是前面的系数；
            # normalization是一个batch中的tgt的token个数，除以这个数会使得每次最终参数更新的时候梯度大小是平均到一个token带来的梯度下降，即最终的梯度与tgt中的token个数无关，是一个标准化的过程
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]  # scores.max(1): 返回scores在第1维的，即每一行的最大值，即所在维度的索引；scores.max(1)[1]: 预测token的索引
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()  # 通过预测结果pred与target中相同位置的idx对比，最终统计出预测正确的token的个数
        num_non_padding = non_padding.sum().item()  # 除了[PAD], ground truth中所有有效token的个数
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # 减2是减去[PAD]与其本身对应的token的位置，保证向量中元素之和为1
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)  # 创建长为tgt_vocab_size的全smoothing_value向量
        one_hot[self.padding_idx] = 0  # one_hot的第一个位置为0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))  # unsqueeze(0)，在第0维增加一个维度，相当于self.one_hot=one_hot.unsqueeze(0)
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # self.one_hot size: 1*tgt_vocab_size，其第一维为长为tgt_vocab_size向量，值为smoothing_value（接近0）
        model_prob = self.one_hot.repeat(target.size(0), 1)  # tensor.size(0): 查看第0维的维度，实际是整个batch的tgt中包含[PAD]，去除起始token的所有token个数; 语句的意思是将self.one_hot的第1维重复target.size(0)次
        # model_prob：size: 整个batch的tgt中包含[PAD]，去除起始token的所有token个数 * tgt_vocab_size(bert词表长度)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)  # 每一行tgt_vocab_size的位置中，target token id对应的位置为self.confidence(接近1，默认0.9)，其余全是smoothing_value（接近0）
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)  # 前面未处理[PAD]，[PAD]对应的行全部置为0，这时KL散度的计算到[PAD]对应的分布（行）时，为0，从而排除了无效token
        # model_prob是经过label smoothing之后的ground truth分布，
        # 第0维等于一个batch中所有tgt groundtruth 中 token个数，包括填充后的[PAD]，
        # model_prob的第一列总为0，因为bert的词表中idx为0的是[PAD]，[PAD]是一个无效字符
        # 每一行对应ground truth token的分类的真实分布，且行之和为1
        # model_prob中[PAD]对应的行全部为0
        return F.kl_div(output, model_prob, reduction='sum')  # F.kl_div的reduction确定输出模式，只有sum才符合后续的计算，后续计算把所有token之间的KL散度平均到了单个token上，F.kl_div的真实分布于拟合分布的行元素的和都为0


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output):
        return {
            "output": output,
            "target": batch.tgt[:,1:],  # 这一块对tgt进行了处理，去除了起始token，因为起始token不参与ground truth的计算，故ground truth为原始tgt行长度-1
        }

    def _compute_loss(self, batch, output, target):
        # batch没有用
        # output: 整体的decoder的output或者shard之后的decoder的output
        # target: 每一行代表一个train example的tgt的ground truth token的bert词表的idx，但经过了_make_shard_state的处理，去掉了起始token的idx（ground truth中不包含起始token）
        # output size: batch_size * len(tgt)-1 * dec_dim  (此batch_size不是args中的batch_size)
        bottled_output = self._bottle(output)  # 将不同example的token合并，用于每个batch计算loss
        # bottled_output: batch_size * len(tgt)-1 * dec_dim -> (batch_size * len(tgt)-1) * dec_dim减少了一个维度
        scores = self.generator(bottled_output)  # scores: out经过generator线性变换，并计算logsoftmax的分布
        # scores的size: (batch_size * (len(tgt)-1)，即总的ground truth的token个数) * tgt_vocab_size: tgt_vocab_size为bert词表的长度
        gtruth = target.contiguous().view(-1)  # gtruth所有tgt的token连在一块，size: 长为batch_size * (len(tgt)-1)的向量

        loss = self.criterion(scores, gtruth)  # KL散度

        stats = self._stats(loss.clone(), scores, gtruth)  # 输入到Statistic对象的loss未对token进行求平均，因为存在梯度累加，需要传入一次报告的总的loss，以及总的token数量

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
