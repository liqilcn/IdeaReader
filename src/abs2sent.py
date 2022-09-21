
import re

import spacy
import pyinflect

nlp = spacy.load("en_core_web_sm")
paper_keywords = [
    # 用于匹配总结性句子，从而进一步缩小范围
    'this paper',
    'this study',
    'this article',
    'this research',
    'this invention',
    'this book',
    'this review',
    'this chapter',
    'this investigation',
    'this work',
    'the paper',
    'the study',
    'the dissertation',
    'the present study',
    'this pioneer study',
    'chapter 8',
]
we_keywords =[
    'we propose',
    'we present',
    'we develop',
    'we built',
    'we examine',
    'we investigate',
    'we introduce',
    'we summarise',
    'we summarize',
    'we explain',
    'we conduct',
    'we demonstrate',
    'we extend',
]
key_verbs = [
        'proposed',
        'developed',
        'introduced',
        'summarised',
        'summarized',
        'built',
        'addressed',
        'offered',
        'presented',
        'provided',
        'conducted',
        'examined',
        'investigated',
        'explained',
        'demonstrated',
        'identified',
        'explored',
        'determineed',
        'studied',
        'analyzed',
        'analysed',
        'understood',
        'provided',
        'contributed',
        'found',
        'assessed',
        'tested',
        'advanced',
        'evaluated',
        'established',
        'revealed',
        'created',
        'explained',
        'verified',
        'conceptualized',
        'measured',
        'enhanced',
        'defined',
        'integrated',
        'reviewed',
        'outlined',
        'described',
        'researched',
        'studied',
        'discussed',
        'answered',
        'concentrated', # 'concentrated on'
        'considered',
        'delved', # 'delves into'
        'revealed',
        'showed',
        'argued',
        'highlighted',
        'applied',
        'synthesized',# 合成？
        'drawed',
        'adopted',
        'dealt', # 'deals with'
        'focused', # 'focuses on'
        'validated',
        'outlined',
        'tackled',
        'figured', # 'figure out'
        'surveyed',
        'reported',
        'gained',
        'indicated',
        'saw',
        'estimated',
        'identified',
        'extended',
        'predicted',
        'interviewed',
        'delineated', # 描述
        'looked', #
        'modeled',
        'elicited', # 引出
    ]
key_verbs1 = [
        'propose',
        'develop',
        'introduce',
        'summarise',
        'summarize',
        'built',
        'address',
        'offer',
        'present',
        'provide',
        'conduct',
        'examine',
        'investigate',
        'explain',
        'demonstrate',
        'identify',
        'explore',
        'determine',
        'study',
        'analyze',
        'analyse',
        'understand',
        'provide',
        'contribute',
        'find',
        'assess',
        'test',
        'advance',
        'evaluate',
        'establish',
        'reveal',
        'create',
        'explain',
        'verify',
        'conceptualize',
        'measure',
        'enhance',
        'define',
        'integrate',
        'review',
        'outline',
        'describe',
        'research',
        'studies',
        'discuss',
        'answer',
        'build',
        'concentrated', # 'concentrated on'
        'concentrat', # 'concentrats on'
        'consider',
        'delves', # 'delves into'
        'reveal',
        'show',
        'argue',
        'highlight',
        'apply',
        'synthesize',# 合成？
        'draw',
        'adopt',
        'deals', # 'deals with'
        'focuses', # 'focuses on'
        'focus', # 'focus on'
        'validate',
        'outline',
        'tackle',
        'theorises',
        'figure', # 'figure out'
        'survey',
        'report',
        'gain',
        'indicate',
        'see',
        'measure',
        'estimate',
        'identifies',
        'extend',
        'predict',
        'interview',
        'delineate', # 描述
        'look', #
        'model',
        'derive',
        'elicit', # 引出
    ]
particular = ['when n identical randomly located nodes']
particular_ans = ['in this paper, the authors analyze the capacity of wireless networks. the authors consider two types of networks, arbitrary networks, where the node locations, destinations of sources, and traffic demands, are all arbitrary, and random networks, where the nodes and their destinations are randomly chosen']
def process_sentence(sent, index):
    sent = sent.lower()
    flag = False
    for word in paper_keywords:
        #print(word)
        if word in sent:
            # sent = past_tense(sent)
            result = sent.replace(word, '[' + str(index+1) + ']')
            flag = True
        else:
            result = sent
        result = result.replace('we', 'the authors')
        result = result.replace('will', '')
        result = result.replace('this', 'the')
        result = result.replace('these', 'the')
        result = result.replace('abstract', '')
        result = result.replace(':', '')
        if flag:
            return result
    for word in key_verbs1:
        #print(word)
        word = 'we ' + word
        if word in sent:
            # sent = past_tense(sent)
            result = sent.replace('we ', '[' + str(index+1) + '] ')
            result = result.replace('will', '')
            result = result.replace('this', 'the')
            result = result.replace('these', 'the')
            result = result.replace('abstract', '')
            result = result.replace(':', '')
            flag = True
        if flag:
            return result
    if flag == False:
        # 首字母大写
        if sent[0].isalpha():
            tem = list(sent)
            tem[0] = tem[0].upper()
            sent = "".join(tem)
        return sent.replace('.', '')+ ' [' + str(index+1) + '].'
def key_sentence(sent, index):
    sent = sent.lower()
    for word in paper_keywords:
        #print(word)
        if word in sent:
            # sent = past_tense(sent)
            if index !=-1:
                result = sent.replace(word, '[' + str(index+1) + ']')
            else:
                result = sent
            result = result.replace('we', 'the authors')
            result = result.replace('will', '')
            result = result.replace('this', 'the')
            result = result.replace('these', 'the')
            result = result.replace('abstract', '')
            result = result.replace(':', '')
            return result
    for word in key_verbs1:
        #print(word)
        word = 'we ' + word
        if word in sent:
            # sent = past_tense(sent)
            if index !=-1:
                result = sent.replace('we ', '[' + str(index+1) + '] ')
            else:
                result = sent.replace('we ', 'this paper ')
            result = result.replace('will', '')
            result = result.replace('this', 'the')
            result = result.replace('these', 'the')
            result = result.replace('abstract', '')
            result = result.replace(':', '')
            return result
    for i, word in enumerate(particular):
        if word in sent:
            # particular_ans[i] = past_tense(particular_ans[i])
            result = particular_ans[i].replace('this paper', '[' + str(index+1) + ']')
            return result
    return ""

def abs2sentence(abs, index):
    key_sents = []
    #sents = abs.split('.')
    sents = re.split("!|\?|\.", abs)
    # print(sents)
    for i, sent in enumerate(sents):
        result = key_sentence(sent, index)

        if result != "":
            # result = past_tense(result)
            if len(key_sents) == 0:
                key_sents.append(result)
                break
            # elif len(key_sents) == 1 and len(key_sents[0])<50:
            #     key_sents.append(result)
    if len(key_sents) == 0:
        return ""
    else:
        return '. '.join(key_sents)+'. '
def cut_text(text):
    sents = text.split('. ')
    new_sents = []
    for i, sent in enumerate(sents):
        sent = sent.replace('refref', '')
        if i==0:
            sent = sent.replace('this paper', 'this part')
            new_sents.append(sent)
        # 只要第一句
        # elif i==1:
        #     if 'paper' not in sent and 'we' not in sent and 'in [' not in sent:
        #         new_sents.append(sent)
        else:
            break
    return '. '.join(new_sents)+'. '
def merge_abs_ext(abs_text, sent_list, id_list=None):
    pre_text = cut_text(abs_text)
    for index, sent in enumerate(sent_list):
        sent = sent.replace('[101]', 'all the authors')
        print("before: ", sent)
        sent = past_tense(sent)
        print("after: ", sent)
        sent = sent.replace('all the authors', '[101]')
        if id_list is None:
            sent = sent.replace('[101]', '['+str(index+1)+']')+' '
        else:
            sent = sent.replace('[101]', '[' + str(id_list[index]+1) + ']')+' '

        pre_text += sent

    return pre_text
def past_tense(text):
    if text == "":
        return ""
    doc_dep = nlp(text)
    for i in range(len(doc_dep)):
        token = doc_dep[i]
        if token.tag_ in ['VBP', 'VBZ'] and token.text not in ['work']:
            # print(token.text, token._.inflect("VBD"))
            if token._.inflect("VBD") is not None:
                text = text.replace(token.text, token._.inflect("VBD"))
    return text
if __name__ == '__main__':
    sent = 'In [100], we study the capacity limitations resulting from the energy supplies in wireless nodes.'
    abs_text = 'when n identical randomly located nodes , each capable of transmitting at w bits per second and using a fixed range , form a wireless network , the throughput / spl lambda / ( n ) obtainable by each node for a randomly chosen destination is / spl theta / ( w / / spl radic / ( nlogn ) ) bits per second under a noninterference protocol . if the nodes are optimally placed in a disk of unit area , traffic patterns are optimally assigned , and each transmissions range is optimally chosen , the bit - distance product that can be transported by the network per second is / spl theta / ( w / spl radic / an ) bit - meters per second . thus even under optimal circumstances , the throughput is only / spl theta / ( w / / spl radic / n ) bits per second for each node for a destination nonvanishingly far away . similar results also hold under an alternate physical model where a required signal - to - interference ratio is specified for successful receptions . fundamentally , it is the need for every node all over the domain to share whatever portion of the channel it is utilizing with nodes in its local neighborhood that is the reason for the constriction in capacity . splitting the channel into several subchannels does not change any of the results . some implications may be worth considering by designers . since the throughput furnished to each user diminishes to zero as the number of users is increased , perhaps networks connecting smaller numbers of users , or featuring connections mostly with nearby neighbors , may be more likely to be find acceptance .'
    abs = 'the wireless channel capacity of a wireless network can be measured by the number of wireless hops connecting each other in the network [0] [1] [2] ? in this paper , we focus on the capacity of wireless networks . we assume that each node is connected to a small number of other nodes that are connected to each other . the topology of the wireless network is determined by the communication range between each node and each other , and the communication rate between any two nodes is the same as that of the other nodes . the communication path between two nodes can be controlled by a constant number of communication channels . for example , a node with a fixed number of neighbors can be used to form a directed acyclic graph ( dag ) , where each node can be connected to all other nodes in the graph , and each wireless channel can be treated as a connected subgraph ( e. a random directed graph ) . each node has a set of neighbors , each of which is connected by some fixed set of neighboring nodes . each neighborhood can be considered as a directed graph ( e. directed acyclic graphs ) , with all other neighbors having the same number of links'
    src_list = [sent]
    #print(key_sentence(sent, index=1))


    sent ="the authors propose a framework for learning convolutional neural networks for arbitrary graphs."
    print(past_tense(sent))

    sent = "Inspired by the fact, the authors propose a recurrent cnn (rcnn) for object recognition by incorporating recurrent connections into each convolutional layer."
    print(past_tense(sent))

    # print(abs2sentence(abs_text, 1))
    #print(merge_abs_ext(abs_text, src_list))