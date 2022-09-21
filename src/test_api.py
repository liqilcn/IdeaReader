

from keywords_es import keyword2papers
import random
from scientific_x_ray_for_survey_generation.scientific_x_ray_for_survey_generation.x_ray_for_survey_generation import pid_list2tree
from graph2json import graph2dic

import pymysql
import argparse
import uvicorn
from test import test_refsum
from models.model_builder import AbsSummarizer
from pytorch_transformers import BertTokenizer
from fastapi import FastAPI, Query, Form, APIRouter, File, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from abs2sent import merge_abs_ext, process_sentence
from sentence_extractor import get_sentence_from_abstract
from utils import *
from word2vec import *

app = FastAPI(
    docs_url='/api/v1/docs',
    redoc_url='/api/v1/redoc',
    openapi_url='/api/v1/openapi.json'
)
router = APIRouter()


db = pymysql.connect(host='xxx',user='xxx',passwd='xxx',db='xxx',port=80)
cursor = db.cursor()

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
# model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")
# model.to(device)
#%%
model_clus = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) # word2vec
def get_paper_title(p_id):
    db = pymysql.connect(host='xxxx', user='xxxx', passwd='xxxx', db='xxxx',
                         port=13306)
    cursor = db.cursor()
    sql = f"SELECT title FROM `am_paper`.`am_paper` WHERE paper_id = {p_id}"
    cursor.execute(sql)
    result = cursor.fetchone()
    if result != None:
        return result[0]
    else:
        return ''

#%%

def get_paper_abstract(p_id):
    db = pymysql.connect(host='xxxx', user='xxxx', passwd='xxxx', db='xxxx',
                         port=13306)
    cursor = db.cursor()
    sql = f"SELECT abstract FROM `am_paper`.`am_paper_abstract` WHERE paper_id = {p_id}"
    cursor.execute(sql)
    result = cursor.fetchone()
    if result != None:
        return result[0]
    else:
        return ''
@router.get('/cluser_paper')
async def cluser_paper(
        input_list: list = Query(..., description='pid', example=['205389644', '407717361', '145726570', '384912289']),
        k: int = Query(..., description='number of para', example=1)
):

    pid2text = {}
    paper_list = [[] for _ in range(k)]
    ori_list, data, pid_list = [], [], []
    for pid in input_list:
        # pid2text[pid] = f'{get_paper_title(pid)} {get_paper_abstract(pid)}'
        pid2text[pid] = f'{get_paper_title(pid)}'
        data.append(get_paper_title(pid))
        pid_list.append(pid)
    # model = train_model(data)
    vecs = train_model2(data, model_clus)
    # %%

    with torch.no_grad():  # 不加这一句，显存会马上爆掉
        pid2embedding = {}
        embs = []
        for i, pid in enumerate(pid_list):
            pid2embedding[pid] = vecs[i].astype('float32')
            embs.append(torch.tensor(pid2embedding[pid]).unsqueeze(0))
            ori_list.append(pid)

        embs = torch.cat(embs, dim=0)
    node2cluster = cluser(embs, k)
    for i, v in enumerate(node2cluster):
        paper_list[v].append(ori_list[i])
    return paper_list
def find_all(source, dest):
     length1,length2 = len(source),len(dest)
     dest_list = []
     temp_list = []
     if length1 < length2:
         return []
     i = 0
     while i <= length1-length2:
         if source[i] == dest[0]:
             dest_list.append(i)
         i += 1
     if dest_list == []:
         return []
     for x in dest_list:
         #print("Now x is:%d. Slice string is :%s"% (x,repr(source[x:x+length2])),end=" ")
         if source[x:x+length2] != dest:
             #print(" dest != slice")
             temp_list.append(x)

     for x in temp_list:
         dest_list.remove(x)
     return dest_list


@router.get('/keyword2related_work2')
async def keyword2related_work2(
        keyword: str = Query(..., description='keyword', example='convolutional neural network'),
):
# def keyword2related_work(keyword):
    papers, src_list, info, sent_dic = keyword2papers(keyword)
    if len(src_list)>5:
        src_list = src_list[:5]
    # papers = papers[:5]
    # src_list = src_list[:5]
    line = test_refsum(src_list, args, checkpoint, model, tokenizer)
    line = clean_text(line, len(src_list))
    return line, papers, info
@router.get('/keyword2meta')
async def keyword2meta(
        keyword: str = Query(..., description='keyword', example='convolutional neural network'),
):
# def keyword2related_work(keyword):
    global abs_list, sents_dic
    sent_dic = []
    papers, src_list, info, sent_dic = keyword2papers(keyword, sent_dic)
    abs_list = src_list
    sents_dic = sent_dic
    return papers, src_list, info
@router.get('/keywords2meta')
async def keywords2meta(
        keywords: list = Query(..., description='keywords', example=['graph neural network', 'contrastive learning']),
):
# def keyword2related_work(keyword):
    global abs_list, sents_dic
    papers_all, src_list_all, info_all, sent_dic_all = [], [], [], []
    for keyword in keywords:
        papers, src_list, info, sent_dic_all = keyword2papers(keyword, sent_dic_all)
        papers_all += papers
        src_list_all += src_list
        info_all += info
    abs_list = src_list_all
    sents_dic = sent_dic_all
    return papers_all, src_list_all, info_all


# @router.post('/abs2related_work2')
# async def abs2related_work2(
#         src_list: str = Form(..., description='text',
#                                 example='["text1", "text2"]')
# ):
@router.get('/abs2related_work2')
async def abs2related_work2(
        id_list: list = Query(..., description='text', example=[1, 2, 3]),
        rule: int = Query(..., description='based on rule', example=1),
):
# @router.get('/abs2related_work2')
# async def abs2related_work2(
#         src_list: list = Query(..., description='text', example=['text1', 'text2', 'text3']),
# ):
    src_list = []
    sent_list = []
    for id in id_list:
        src_list.append(abs_list[int(id)])
        if rule == 1:
            sent_list.append(sents_dic[int(id)])
        else:
            sent = get_sentence_from_abstract(abs_list[int(id)])
            sent = process_sentence(sent, 100)
            sent_list.append(sent)
    print(src_list)
    line = test_refsum(src_list, args, checkpoint, model, tokenizer)
    line = clean_text(line, len(src_list))
    text = merge_abs_ext(line, sent_list)

    return text
@router.get('/pid_list2skeleton_tree')
async def pid_list2skeleton_tree(
        keywords: list = Query(..., description='keywords', example=['graph neural network', 'contrastive learning']),
        pid_list: list = Query(..., description='pid', example=['205389644', '407717361', '145726570', '384912289']),
):
    keywords_str = '& '.join(keywords)
    tree = pid_list2tree(pid_list)
    dic = graph2dic(tree, keywords_str)
    return dic
@router.get('/pid2related_work2')
async def pid2related_work2(
        pid: list = Query(..., description='pid', example=['205389644', '407717361', '145726570', '384912289']),
):
    a = pid
    src_list = []
    for id in pid:
        sql = "SELECT abstract FROM am_paper_abstract WHERE paper_id=%s" % (id)

        cursor.execute(sql)
        results = cursor.fetchall()
        try:
            src_list.append(results[0][0])
        except:
            pass

    line = test_refsum(src_list, args, checkpoint, model, tokenizer)
    line = clean_text(line, len(src_list))
    return line
def pid2related_work(pid, sent_list, id_list=None):
    src_list = []
    for id in pid:
        sql = "SELECT abstract FROM am_paper_abstract WHERE paper_id=%s" % (id)

        cursor.execute(sql)
        results = cursor.fetchall()
        try:
            src_list.append(results[0][0])
        except:
            pass

    line = test_refsum(src_list, args, checkpoint, model, tokenizer)
    line = clean_text(line, len(src_list), id_list)
    text = merge_abs_ext(line, sent_list, id_list)
    return text


# keyword = 'graph classification'
# keyword2related_work(keyword)
# async def catch_exceptions_middleware(request: Request, call_next):
#     try:
#         return await call_next(request)
#     except Exception as e:
#         traceback.print_exc()
#         return Response("Internal server error " + str(e), status_code=500)
#
# app.middleware('http')(catch_exceptions_middleware)
app.include_router(router, prefix='/api/v1')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# text='the problem of coalition structure generation has been extensively studied in the literature refref refref refref . the most relevant work is refref refref , where the goal is to partition the set of agents into disjoint coalitionsjoint coalitions . in refref , the problem is defined as : given a set of disjoint coalitions , find an optimal coalition structure that maximizes the sum of the total payoff of the agents . in , the authors propose a heuristic algorithm for choosing the best coalition structure . the heuristic algorithm is based on the notion of a heuristic function that maps the agents into disjunctions ofreff . inrefff'
def clean_text(text, mx_length, id_list=None):
    sents = text.split('.')

    new_sents = []

    if id_list is None:
        start_id = 0
    else:
        if len(id_list) == 0:
            return ""
        start_id = id_list[0]
        mx_length = id_list[-1]+1
    pos = start_id
    flag = False
    for sent in sents:
        indexs = find_all(sent, 'refref ')
        # print(indexs)
        if len(indexs)>1:
            for i in range(mx_length):
                if 'refref ' in sent:
                    if flag==False:
                        # sent = sent.replace('refref', '['+str(pos+1)+']', 1)
                        sent = sent.replace('refref ', '[' + str(pos + 1) + '-' + str(mx_length) + '] ', 1)
                        break
                        # if pos < mx_length-1:
                        #     pos += 1
                        # else:
                        #     flag = True
                        #     pos = random.randint(start_id, mx_length-1)
                    else:
                        pos = random.randint(start_id, mx_length - 1)
                        # sent = sent.replace('refref', '[' + str(pos+1) + ']', 1)
                        sent = sent.replace('refref ', '', 1)
                else:
                    break
            for i in range(mx_length):
                sent = sent.replace('refref ', '', 1)
        elif len(indexs)==1 :
            if flag==False:
                sent = sent.replace('refref ', '[' + str(pos + 1) + '-' + str(mx_length) + '] ', 1)
                # if pos<mx_length-1:
                #     sent = sent.replace('refref', '[' + str(pos+1) + ']', 1)
                #     pos+=1
                # else:
                #     sent = sent.replace('refref', '[' + str(pos+1) + ']', 1)
            else:
                pos = random.randint(0, mx_length-1)
                sent = sent.replace('refref', '[' + str(pos+1) + ']', 1)

        if len(sent)>10:
            new_sents.append(sent)
    # print('.'.join(new_sents))
    for i, sent in enumerate(new_sents):
        new_sents[i] = new_sents[i].strip()
        if new_sents[i][0].isalpha():
            tem = list(new_sents[i])
            tem[0] = tem[0].upper()
            new_sents[i] = "".join(tem)

    return '. '.join(new_sents)

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
parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-load_from_extractive", default='', type=str)

parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-lr_bert", default=2e-3, type=float)
parser.add_argument("-lr_dec", default=2e-3, type=float)
parser.add_argument("-use_bert_emb", type=str2bool, nargs='?', const=True, default=True)

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
parser.add_argument("-alpha", default=0.95, type=float)
parser.add_argument("-beam_size", default=5, type=int)
parser.add_argument("-min_length", default=40, type=int)
parser.add_argument("-max_length", default=80, type=int)
parser.add_argument("-max_tgt_len", default=80, type=int)

parser.add_argument("-param_init", default=0, type=float)
parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-optim", default='adam', type=str)
parser.add_argument("-lr", default=1, type=float)
parser.add_argument("-beta1", default=0.9, type=float)
parser.add_argument("-beta2", default=0.999, type=float)
parser.add_argument("-warmup_steps", default=8000, type=int)
parser.add_argument("-warmup_steps_bert", default=8000, type=int)
parser.add_argument("-warmup_steps_dec", default=8000, type=int)
parser.add_argument("-max_grad_norm", default=0, type=float)

parser.add_argument("-save_checkpoint_steps", default=5, type=int)
parser.add_argument("-accum_count", default=1, type=int)
parser.add_argument("-report_every", default=1, type=int)
parser.add_argument("-train_steps", default=1000, type=int)
parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)
parser.add_argument('-log_file', default='../logs/abs_bert_delve')
parser.add_argument('-seed', default=666, type=int)

parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-test_from", default='../models/bertsumabs.pt')
parser.add_argument("-test_start_from", default=-1, type=int)

parser.add_argument("-train_from", default='')
parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

args = parser.parse_args()
args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
args.world_size = len(args.gpu_ranks)

print("Loading checkpoint")
checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
print("Finish Loading checkpoint")
device = "cpu" if args.visible_gpus == '-1' else "cuda"
model = AbsSummarizer(args, device, checkpoint)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
abs_list = []
sents_dic = {}
if __name__ == '__main__':

    src_list = [
        'this paper proposes a vehicletovehicle communication protocol for cooperative collision warning . emerging wireless technologies for vehicletovehicle ( v2v ) and vehicletoroadside ( v2r ) communications such as dsrc [1] are promising to dramatically reduce the number of fatal roadway accidents by providing early warnings . one major technical challenge addressed in this paper is to achieve lowlatency in delivering emergency warnings in various road situations . based on a careful analysis of application requirements , we design an effective protocol , comprising congestion control policies , service differentiation mechanisms and methods for emergency warning dissemination . simulation results demonstrate that the proposed protocol achieves low latency in delivering emergency warnings and efficient bandwidth usage in stressful road scenarios .', \
        'in this paper we conduct a feasibility study of delaycritical safety applications over vehicular ad hoc networks based on the emerging dedicated short range communications ( dsrc ) standard . in particular , we quantify the bit error rate , throughput and latency associated with vehicle collision avoidance applications running on top of mobile ad hoc networks employing the physical and mac layers of dsrc . towards this objective , the study goes through two phases . first , we conduct a detailed simulation study of the dsrc physical layer in order to judge the link bit error rate performance under a wide variety of vehicles speeds and multipath delay spreads . we observe that the physical layer is highly immune to large delay spreads that might arise in the highway environment whereas performance degrades considerably at high speeds in a multipath environment . second , we develop a simulation testbed for a dsrc vehicular ad hoc network executing vehicle collision avoidance applications in an attempt to gauge the level of support the dsrc standard provides for this type of applications . initial results reveal that dsrc achieves promising latency performance , yet , the throughput performance needs further improvement .', \
        'this paper studies the design of layer2 protocols for a vehicle to send safety messages to other vehicles . the target is to send vehicle safety messages with high reliability and low delay . the communication is onetomany , local , and geosignificant . the vehicular communication network is adhoc , highly mobile , and with large numbers of contending nodes . the messages are very short , have a brief useful lifetime , but must be received with high probability . for this environment , this paper explores the efficacy of rapid repetition of broadcast messages . this paper proposes several random access protocols for medium access control . the protocols are compatible with the dedicated short range communications ( dsrc ) multichannel architecture . analytical bounds on performance of the proposed protocols are derived . simulations are conducted to assess the reception reliability and channel usage of the protocols . the sensitivity of the protocol performance is evaluated under various offered traffic and vehicular traffic flows . the results show our approach is feasible for vehicle safety messages in dsrc .']
    src_list = [
        "the task of mining association rules consists of two main steps . the first involves finding the set of all frequent itemsets . the second step involves testing and generating all high confidence rules among itemsets . in this paper we show that it is not necessary to mine all frequent itemsets in the first step , instead it is sufficient to mine the set of closed frequent itemsets , which is much smaller than the set of all frequent itemsets . it is also not necessary to mine the set of all possible rules . we show that any rule between itemsets is equivalent to some rule between closed itemsets . thus many redundant rules can be eliminated . furthermore , we present charm , an efficient algorithm for mining all closed frequent itemsets . an extensive experimental evaluation on a number of real and synthetic databases shows that charm outperforms previous methods by an order of magnitude or more . it is also linearly scalable in the number of transactions and the number of closed itemsets found .",
        "mining frequent closed itemsets provides complete and nonredundant results for frequent pattern analysis . extensive studies have proposed various strategies for efficient frequent closed itemset mining , such as depthfirst search vs. breadthfirst search , vertical formats vs. horizontal formats , treestructure vs. other data structures , topdown vs. bottomup traversal , pseudo projection vs. physical projection of conditional database , etc . it is the right time to ask what are the pros and cons of the strategies ? and what and how can we pick and integrate the best strategies to achieve higher performance in general cases ? in this study , we answer the above questions by a systematic study of the search strategies and develop a winning algorithm closet . closet integrates the advantages of the previously proposed effective strategies as well as some ones newly developed here . a thorough performance study on synthetic and real data sets has shown the advantages of the strategies and the improvement of closet over existing mining algorithms , including closet , charm and op , in terms of runtime , memory usage and scalability .",
        "the growth of bioinformatics has resulted in datasets with new characteristics . these datasets typically contain a large number of columns and a small number of rows . for example , many gene expression datasets may contain 10,000100,000 columns but only 1001000 rows . such datasets pose a great challenge for existing ( closed ) frequent pattern discovery algorithms , since they have an exponential dependence on the average row length . in this paper , we describe a new algorithm called carpenter that is specially designed to handle datasets having a large number of attributes and relatively small number of rows . several experiments on real bioinformatics datasets show that carpenter is orders of magnitude better than previous closed pattern mining algorithms like closet and charm ."]
    src_list = [
        "the basal ganglia support learning to exploit decisions that have yielded positive outcomes in the past . in contrast , limited evidence implicates the prefrontal cortex in the process of making strategic exploratory decisions when the magnitude of potential outcomes is unknown . here we examine neurogenetic contributions to individual differences in these distinct aspects of motivated .d human behavior , using a temporal decisionmaking task and computational analysis . we show that two genes controlling striatal rve dopamine function , darpp32 ( also called ppp1r1b ) and drd2 , are associated with exploitative learning to adjust response se times incrementally as a function of positive and negative decision outcomes . in contrast , a gene primarily controlling prefrontal re dopamine function ( comt ) is associated with a particular type of 'directed exploration ' , in which exploratory decisions are made ts in proportion to bayesian uncertainty about whether other choices might produce outcomes that are better than the status quo . igh quantitative model fits reveal that genetic factors modulate independent parameters of a reinforcement learning system . r l l a",
        "one prevalent theory of learning states that dopamine neurons signal mismatches between expected and actual outcomes , called temporal difference errors ( tdes ) . evidence indicates that dopamine system dysfunction is involved in negative symptoms of schizophrenia ( sz ) , including avolition and anhedonia . as such , we predicted that brain responses to tdes in dopamine midbrain nuclei and target areas would be abnormal in sz . a total of 18 clinically stable patients with chronic sz and 18 controls participated in an fmri study , which used a passive conditioning task . in the task , the delivery of a small amount of juice followed a light stimulus by exactly 6 s on approximately 75 of 78 total trials , and was further delayed by 47 s on the remaining trials . the delayed juice delivery was designed to elicit the two types of tde signals , associated with the recognition that a reward was omitted at the expected time , and delivered at an unexpected time . main effects of tde valence and group differences in the positivenegative tde contrast ( unexpected juice deliveriesjuice omissions ) were assessed through wholebrain and regions of interest ( roi ) analyses . main effects of tde valence were observed for the entire sample in the midbrain , left putamen , left cerebellum , and primary gustatory cortex , bilaterally . wholebrain analyses revealed group differences in the positivenegative tde contrast in the right putamen and left precentral gyrus , whereas roi analyses revealed additional group differences in the midbrain , insula , and parietal operculum , on the right , the putamen and cerebellum , on the left , and the frontal operculum , bilaterally . further , these group differences were generally driven by attenuated responses in patients to positive tdes ( unexpected juice deliveries ) , whereas responses to negative tdes ( unexpected juice omissions ) were largely intact . patients also showed reductions in responses to juice deliveries on standard trials , and more blunted reinforcer responses in the left putamen corresponded to higher ratings of avolition . these results provide evidence that sz patients show abnormal brain responses associated with the processing of a primary reinforcer , which may be a source of motivational deficits . âš 2009 nature publishing group .",
        "background : rewards and punishments may make distinct contributions to learning via separate striatalcortical pathways . we investigated whether frontostriatal dysfunction in schizophrenia ( sz ) is characterized by selective impairment in either reward ( go ) or punishmentdriven ( nogo ) learning . methods : we administered two versions of a probabilistic selection task to 40 schizophrenia patients and 31 control subjects , using difficult to verbalize stimuli ( experiment 1 ) and nameable objects ( experiment 2 ) . in an acquisition phase , participants learned to choose between three different stimulus pairs ( ab , cd , ef ) presented in random order , based on probabilistic feedback ( 80 , 70 , 60 ) . we used analyses of variance ( anovas ) to assess the effects of group and reinforcement probability on two measures of contingency learning . to characterize the preference of subjects for choosing the most rewarded stimulus and avoiding the most punished stimulus , we subsequently tested participants with novel pairs of stimuli involving either a or b , providing no feedback . results : control subjects demonstrated superior performance during the first 40 acquisition trials in each of the 80 and 70 conditions versus the 60 condition ; patients showed similarly impaired ( 60 ) performance in all three conditions . in novel test pairs , patients showed decreased preference for the most rewarded stimulus ( a ; t 2.674 ; p .01 ) . patients were unimpaired at avoiding the most negative stimulus ( b ; t .737 ) . conclusions : the results of these experiments provide additional evidence for the presence of deficits in reinforcement learning in sz , suggesting that rewarddriven learning may be more profoundly impaired than punishmentdriven learning ."]
    src_list = [
        "social networks are growing in number and size , with hundreds of millions of user accounts among them . one added benefit of these networks is that they allow users to encode more information about their relationships than just stating who they know . in this work , we are particularly interested in trust relationships , and how they can be used in designing interfaces . in this paper , we present filmtrust , a website that uses trust in webbased social networks to create predictive movie recommendations . using the filmtrust system as a foundation , we show that these recommendations are more accurate than other techniques when the user 's opinions about a film are divergent from the average . we discuss this technique both as an application of social network analysis , as well as how it suggests other analyses that can be performed to help improve collaborative filtering algorithms of all types .",
        "although recommender systems have been comprehensively analyzed in the past decade , the study of socialbased recommender systems just started . in this paper , aiming at providing a general method for improving recommender systems by incorporating social network information , we propose a matrix factorization framework with social regularization . the contributions of this paper are fourfold : ( 1 ) we elaborate how social network information can benefit recommender systems ; ( 2 ) we interpret the differences between socialbased recommender systems and trustaware recommender systems ; ( 3 ) we coin the term social regularization to represent the social constraints on recommender systems , and we systematically illustrate how to design a matrix factorization objective function with social regularization ; and ( 4 ) the proposed method is quite general , which can be easily extended to incorporate other contextual information , like social tags , etc . the empirical analysis on two large datasets demonstrates that our approaches outperform other stateoftheart methods .",
        "social media treats all users the same : trusted friend or total stranger , with little or nothing in between . in reality , relationships fall everywhere along this spectrum , a topic social science has investigated for decades under the theme of tie strength . our work bridges this gap between theory and practice . in this paper , we present a predictive model that maps social media data to tie strength . the model builds on a dataset of over 2,000 social media ties and performs quite well , distinguishing between strong and weak ties with over 85 accuracy . we complement these quantitative findings with interviews that unpack the relationships we could not predict . the paper concludes by illustrating how modeling tie strength can improve social media design elements , including privacy controls , message routing , friend introductions and information prioritization ."]
    src_list = [
        "evaluative texts on the web have become a valuable source of opinions on products , services , events , individuals , etc . recently , many researchers have studied such opinion sources as product reviews , forum posts , and blogs . however , existing research has been focused on classification and summarization of opinions using natural language processing and data mining techniques . an important issue that has been neglected so far is opinion spam or trustworthiness of online opinions . in this paper , we study this issue in the context of product reviews , which are opinion rich and are widely used by consumers and product manufacturers . in the past two years , several startup companies also appeared which aggregate opinions from product reviews . it is thus high time to study spam in reviews . to the best of our knowledge , there is still no published study on this topic , although web spam and email spam have been investigated extensively . we will see that opinion spam is quite different from web spam and email spam , and thus requires different detection techniques . based on the analysis of 5.8 million reviews and 2.14 million reviewers from amazon.com , we show that opinion spam in reviews is widespread . this paper analyzes such spam activities and presents some novel techniques to detect them .",
        "in recent years , opinion mining attracted a great deal of research attention . however , limited work has been done on detecting opinion spam ( or fake reviews ) . the problem is analogous to spam in web search [ 1 , 9 11 ] . however , review spam is harder to detect because it is very hard , if not impossible , to recognize fake reviews by manually reading them [2] . this paper deals with a restricted problem , i.e . identifying unusual review patterns which can represent suspicious behaviors of reviewers . we formulate the problem as finding unexpected rules . the technique is domain independent . using the technique , we analyzed an amazon.com review dataset and found many unexpected rules and rule groups which indicate spam activities .",
        "consumers increasingly rate , review and research products online ( jansen , 2010 ; litvin et al . 2008 ) . consequently , websites containing consumer reviews are becoming targets of opinion spam . while recent work has focused primarily on manually identifiable instances of opinion spam , in this work we study deceptive opinion spamfictitious opinions that have been deliberately written to sound authentic . integrating work from psychology and computational linguistics , we develop and compare three approaches to detecting deceptive opinion spam , and ultimately develop a classifier that is nearly 90 accurate on our goldstandard opinion spam dataset . based on feature analysis of our learned models , we additionally make several theoretical contributions , including revealing a relationship between deceptive opinions and imaginative writing .",
        "link to online http : www.csi.ucd.iecontentdistortionvalidationcriterionversion identificationsuspiciousreviews some rights reserved . for more information , please see the item record link above ."]
    src_list = [
        "this paper presents an approach to automatically optimizing the retrieval quality of search engines using clickthrough data . intuitively , a good information retrieval system should present relevant documents high in the ranking , with less relevant documents following below . while previous approaches to learning retrieval functions from examples exist , they typically require training data generated from relevance judgments by experts . this makes them difficult and expensive to apply . the goal of this paper is to develop a method that utilizes clickthrough data for training , namely the querylog of the search engine in connection with the log of links the users clicked on in the presented ranking . such clickthrough data is available in abundance and can be recorded at very low cost . taking a support vector machine ( svm ) approach , this paper presents a method for learning retrieval functions . from a theoretical perspective , this method is shown to be wellfounded in a risk minimization framework . furthermore , it is shown to be feasible even for large sets of queries and features . the theoretical results are verified in a controlled experiment . it shows that the method can effectively adapt the retrieval function of a metasearch engine to a particular group of users , outperforming google in terms of retrieval quality after only a couple of hundred training examples .",
        "we investigate using gradient descent methods for learning ranking functions ; we propose a simple probabilistic cost function , and we introduce ranknet , an implementation of these ideas using a neural network to model the underlying ranking function . we present test results on toy data and on data from a commercial internet search engine .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach .",
        "the paper is concerned with learning to rank , which is to construct a model or a function for ranking objects . learning to rank is useful for document retrieval , collaborative filtering , and many other applications . several methods for learning to rank have been proposed , which take object pairs as 'instances ' in learning . we refer to them as the pairwise approach in this paper . although the pairwise approach offers advantages , it ignores the fact that ranking is a prediction task on list of objects . the paper postulates that learning to rank should adopt the listwise approach in which lists of objects are used as 'instances ' in learning . the paper proposes a new probabilistic method for the approach . specifically it introduces two probability models , respectively referred to as permutation probability and top k probability , to define a listwise loss function for learning . neural network and gradient descent are then employed as model and algorithm in the learning method . experimental results on information retrieval show that the proposed listwise approach performs better than the pairwise approach ."]
    # print(pid2related_work(['205389644', '407717361', '145726570', '384912289']))
    print()
    uvicorn.run(app=app, host="0.0.0.0", port=9921, workers=1)
