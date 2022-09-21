#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext
from prepare_test_data import prepare_data
import  json

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def test_refsum(src_list, args=None, checkpoint=None, model=None, tokenizer=None):

    prepare_data(src_list)
    if args == None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
        parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
        parser.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test'])
        parser.add_argument("-bert_data_path", default='../bert_data_test/delve')
        parser.add_argument("-model_path", default='../models/')
        parser.add_argument("-result_path", default='../results/delve')
        parser.add_argument("-temp_dir", default='../temp')

        parser.add_argument("-batch_size", default=3000, type=int)
        parser.add_argument("-test_batch_size", default=500, type=int)

        parser.add_argument("-max_pos", default=512, type=int)
        parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
        parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
        parser.add_argument("-load_from_extractive", default='', type=str)

        parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=True)
        parser.add_argument("-lr_bert", default=2e-3, type=float)
        parser.add_argument("-lr_dec", default=2e-3, type=float)
        parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=True)

        parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("-dec_dropout", default=0.2, type=float)
        parser.add_argument("-dec_layers", default=6, type=int)
        parser.add_argument("-dec_hidden_size", default=768, type=int)
        parser.add_argument("-dec_heads", default=8, type=int)
        parser.add_argument("-dec_ff_size", default=2048, type=int)
        parser.add_argument("-enc_hidden_size", default=512, type=int)
        parser.add_argument("-enc_ff_size", default=512, type=int)
        parser.add_argument("-enc_dropout", default=0.2, type=float)
        parser.add_argument("-enc_layers", default=6, type=int)

        # params for EXT
        parser.add_argument("-ext_dropout", default=0.2, type=float)
        parser.add_argument("-ext_layers", default=2, type=int)
        parser.add_argument("-ext_hidden_size", default=768, type=int)
        parser.add_argument("-ext_heads", default=8, type=int)
        parser.add_argument("-ext_ff_size", default=2048, type=int)

        parser.add_argument("-label_smoothing", default=0.1, type=float)
        parser.add_argument("-generator_shard_size", default=32, type=int)
        parser.add_argument("-alpha",  default=0.95, type=float)
        parser.add_argument("-beam_size", default=5, type=int)
        # parser.add_argument("-min_length", default=200, type=int)
        # parser.add_argument("-max_length", default=300, type=int)
        # parser.add_argument("-max_tgt_len", default=300, type=int)
        parser.add_argument("-min_length", default=500, type=int)
        parser.add_argument("-max_length", default=800, type=int)
        parser.add_argument("-max_tgt_len", default=800, type=int)



        parser.add_argument("-param_init", default=0, type=float)
        parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
        parser.add_argument("-optim", default='adam', type=str)
        parser.add_argument("-lr", default=1, type=float)
        parser.add_argument("-beta1", default= 0.9, type=float)
        parser.add_argument("-beta2", default=0.999, type=float)
        parser.add_argument("-warmup_steps", default=8000, type=int)
        parser.add_argument("-warmup_steps_bert", default=8000, type=int)
        parser.add_argument("-warmup_steps_dec", default=8000, type=int)
        parser.add_argument("-max_grad_norm", default=0, type=float)

        parser.add_argument("-save_checkpoint_steps", default=5, type=int)
        parser.add_argument("-accum_count", default=1, type=int)
        parser.add_argument("-report_every", default=1, type=int)
        parser.add_argument("-train_steps", default=1000, type=int)
        parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


        parser.add_argument('-visible_gpus', default='-1', type=str)
        parser.add_argument('-gpu_ranks', default='0', type=str)
        parser.add_argument('-log_file', default='../logs/abs_bert_delve')
        parser.add_argument('-seed', default=666, type=int)

        parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
        parser.add_argument("-test_from", default='../models/bertsumabs.pt')
        parser.add_argument("-test_start_from", default=-1, type=int)

        parser.add_argument("-train_from", default='')
        parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
        parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

        args = parser.parse_args()
        args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
        args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, device_id)
        elif (args.mode == 'validate'):
            validate_abs(args, device_id)
        elif (args.mode == 'lead'):
            baseline(args, cal_lead=True)
        elif (args.mode == 'oracle'):
            baseline(args, cal_oracle=True)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step, checkpoint, model, tokenizer)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)

    elif (args.task == 'ext'):
        if (args.mode == 'train'):
            train_ext(args, device_id)
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
                test_text_abs(args, device_id, cp, step)

    with open("../results/delve.600000.candidate", 'r', encoding='utf8') as f:
        data = f.readlines()
        for line in data:
            print(line)
            return line
if __name__ == '__main__':
    src_list = [
            "evaluative texts on the web have become a valuable source of opinions on products , services , events , individuals , etc . recently , many researchers have studied such opinion sources as product reviews , forum posts , and blogs . however , existing research has been focused on classification and summarization of opinions using natural language processing and data mining techniques . an important issue that has been neglected so far is opinion spam or trustworthiness of online opinions . in this paper , we study this issue in the context of product reviews , which are opinion rich and are widely used by consumers and product manufacturers . in the past two years , several startup companies also appeared which aggregate opinions from product reviews . it is thus high time to study spam in reviews . to the best of our knowledge , there is still no published study on this topic , although web spam and email spam have been investigated extensively . we will see that opinion spam is quite different from web spam and email spam , and thus requires different detection techniques . based on the analysis of 5.8 million reviews and 2.14 million reviewers from amazon.com , we show that opinion spam in reviews is widespread . this paper analyzes such spam activities and presents some novel techniques to detect them .",
            "in recent years , opinion mining attracted a great deal of research attention . however , limited work has been done on detecting opinion spam ( or fake reviews ) . the problem is analogous to spam in web search [ 1 , 9 11 ] . however , review spam is harder to detect because it is very hard , if not impossible , to recognize fake reviews by manually reading them [2] . this paper deals with a restricted problem , i.e . identifying unusual review patterns which can represent suspicious behaviors of reviewers . we formulate the problem as finding unexpected rules . the technique is domain independent . using the technique , we analyzed an amazon.com review dataset and found many unexpected rules and rule groups which indicate spam activities .",
            "consumers increasingly rate , review and research products online ( jansen , 2010 ; litvin et al . 2008 ) . consequently , websites containing consumer reviews are becoming targets of opinion spam . while recent work has focused primarily on manually identifiable instances of opinion spam , in this work we study deceptive opinion spamfictitious opinions that have been deliberately written to sound authentic . integrating work from psychology and computational linguistics , we develop and compare three approaches to detecting deceptive opinion spam , and ultimately develop a classifier that is nearly 90 accurate on our goldstandard opinion spam dataset . based on feature analysis of our learned models , we additionally make several theoretical contributions , including revealing a relationship between deceptive opinions and imaginative writing .",
            "link to online http : www.csi.ucd.iecontentdistortionvalidationcriterionversion identificationsuspiciousreviews some rights reserved . for more information , please see the item record link above ."]
    src_list = [
        "social networks are growing in number and size , with hundreds of millions of user accounts among them . one added benefit of these networks is that they allow users to encode more information about their relationships than just stating who they know . in this work , we are particularly interested in trust relationships , and how they can be used in designing interfaces . in this paper , we present filmtrust , a website that uses trust in webbased social networks to create predictive movie recommendations . using the filmtrust system as a foundation , we show that these recommendations are more accurate than other techniques when the user 's opinions about a film are divergent from the average . we discuss this technique both as an application of social network analysis , as well as how it suggests other analyses that can be performed to help improve collaborative filtering algorithms of all types .",
        "although recommender systems have been comprehensively analyzed in the past decade , the study of socialbased recommender systems just started . in this paper , aiming at providing a general method for improving recommender systems by incorporating social network information , we propose a matrix factorization framework with social regularization . the contributions of this paper are fourfold : ( 1 ) we elaborate how social network information can benefit recommender systems ; ( 2 ) we interpret the differences between socialbased recommender systems and trustaware recommender systems ; ( 3 ) we coin the term social regularization to represent the social constraints on recommender systems , and we systematically illustrate how to design a matrix factorization objective function with social regularization ; and ( 4 ) the proposed method is quite general , which can be easily extended to incorporate other contextual information , like social tags , etc . the empirical analysis on two large datasets demonstrates that our approaches outperform other stateoftheart methods .",
        "social media treats all users the same : trusted friend or total stranger , with little or nothing in between . in reality , relationships fall everywhere along this spectrum , a topic social science has investigated for decades under the theme of tie strength . our work bridges this gap between theory and practice . in this paper , we present a predictive model that maps social media data to tie strength . the model builds on a dataset of over 2,000 social media ties and performs quite well , distinguishing between strong and weak ties with over 85 accuracy . we complement these quantitative findings with interviews that unpack the relationships we could not predict . the paper concludes by illustrating how modeling tie strength can improve social media design elements , including privacy controls , message routing , friend introductions and information prioritization ."]
    src_list = [
        "this paper presents an approach to automatically optimizing the retrieval quality of search engines using clickthrough data . intuitively , a good information retrieval system should present relevant documents high in the ranking , with less relevant documents following below . while previous approaches to learning retrieval functions from examples exist , they typically require training data generated from relevance judgments by experts . this makes them difficult and expensive to apply . the goal of this paper is to develop a method that utilizes clickthrough data for training , namely the querylog of the search engine in connection with the log of links the users clicked on in the presented ranking . such clickthrough data is available in abundance and can be recorded at very low cost . taking a support vector machine ( svm ) approach , this paper presents a method for learning retrieval functions . from a theoretical perspective , this method is shown to be wellfounded in a risk minimization framework . furthermore , it is shown to be feasible even for large sets of queries and features . the theoretical results are verified in a controlled experiment . it shows that the method can effectively adapt the retrieval function of a metasearch engine to a particular group of users , outperforming google in terms of retrieval quality after only a couple of hundred training examples .",
        "we investigate using gradient descent methods for learning ranking functions ; we propose a simple probabilistic cost function , and we introduce ranknet , an implementation of these ideas using a neural network to model the underlying ranking function . we present test results on toy data and on data from a commercial internet search engine .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach ."]

    with open('tree/sentence/text(1).json', 'r', encoding='utf8') as f:
        dic = json.load(f)
    results = {}
    for key in dic:
        text = test_refsum(dic[key])
        results[key] = text
    with open('tree/sentence/result.json', 'w', encoding='utf8') as f:
        js_str = json.dumps(results)
        f.write(js_str)
