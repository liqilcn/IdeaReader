import os

from nltk import sent_tokenize
import pandas as pd
import torch
import torch.nn as nn
# from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
try:
    from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
except:
    from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = "allenai/scibert_scivocab_uncased"
        self.encoder = AutoModelForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label=None):
        res = self.encoder(text, labels=label)[:2]
        return res


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def get_scores(model, tree_loader, device):
    scores = []
    model.eval()
    with torch.no_grad():
        for text, _ in tree_loader:
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text)
            output = output[0]
            score = output[:, 1].tolist()
            scores.extend(score)

    return scores


def get_sentence_from_abstract(abstract: str, model_path='../models/scibert_ext.pt'):
    # split abstract to sentences
    abstract_sentences = sent_tokenize(abstract)
    abstract_sentences = [bytes(s, 'utf-8').decode('utf-8', 'ignore') for s in abstract_sentences]
    abstract_sentences = [s.replace('\0', '') for s in abstract_sentences]
    df = pd.DataFrame(abstract_sentences, columns=["text"])
    # tmp file
    tmp_file_path = "./abstract_sentences.csv"
    df.to_csv(tmp_file_path, index=False)
    # load sentences
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

    MAX_SEQ_LEN = 64
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                       fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('text', text_field), (None, None)]
    input_data = TabularDataset(path=tmp_file_path, format='CSV', fields=fields, skip_header=True)
    input_iter = Iterator(input_data, batch_size=16, device=device, train=False, shuffle=False, sort=False)
    # delete tmp file
    os.remove(tmp_file_path)
    # get scores
    best_model = BERT().to(device)
    load_checkpoint(model_path, best_model, device)
    scores = get_scores(best_model, input_iter, device)
    return abstract_sentences[scores.index(max(scores))]


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    abstract = 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5 (7.7 point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).'
    model_path = '../models/scibert_ext.pt'
    topic_sentence = get_sentence_from_abstract(abstract, model_path)
    print(topic_sentence)
