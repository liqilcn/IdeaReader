import json
import os
from os import path
import pymysql
from abs2sent import abs2sentence, past_tense, process_sentence
from test_api import pid2related_work
from reference import get_papers_formated_reference
from sentence_extractor import get_sentence_from_abstract

db = pymysql.connect(host='xxx',user='xxx',passwd='xxx',db='xxx',port=80)
cursor = db.cursor()
def sent_upper(sent):
    sent = sent.strip()
    if sent[0].isalpha():
        tem = list(sent)
        tem[0] = tem[0].upper()
        sent = "".join(tem)
    return sent
def load_file(branch_path):
    with open(branch_path, 'r', encoding='utf8') as f:
        branch_dic = json.load(f)
    return branch_dic
def pid2abs(pid):
    sql = 'SELECT abstract FROM am_paper_abstract WHERE paper_id=%s' % (pid)

    cursor.execute(sql)
    results = cursor.fetchall()
    try:
        abs = results[0][0]
        return abs
    except:
        return ''

def pid2title(pid):
    sql = 'SELECT title FROM am_paper WHERE paper_id=%s' % (pid)
    cursor.execute(sql)
    results = cursor.fetchall()
    try:
        title = results[0][0]
        return title
    except:
        return ''
def pro_branch(branch_dic, rule=True):
    global results
    name = branch_dic['clusterNames']
    results['tracing']['topics'] = name
    branches = branch_dic['branches']
    cnt_glob = 0
    topic_weights = []
    text_list, ref_list = [], []
    mx_weight = 0
    for i, branch in enumerate(branches):
        pid2score = {}

        for dic in branch:
            pid2score[dic['paper_id']] = dic['score']
        branch_name = name[i]
        results['tracing']['reference'][branch_name] = []
        sent_list, id_list = [], []
        pid_list = []
        title_list = []
        print(pid2score)
        sorted_list = sorted(pid2score.items(), key=lambda x: x[1], reverse=True)
        cnt = 1
        score_topic = 0
        for i, item in enumerate(sorted_list):
            pid, score = item[0], item[1]
            score_topic += score
            abs = pid2abs(pid)
            if abs != '' and len(pid_list)<5:
                if rule:
                    sent = abs2sentence(abs, 100)
                else:
                    sent = get_sentence_from_abstract(abs)
                    sent = process_sentence(sent, 100)

                if sent != '':

                    title = pid2title(pid)
                    # if cnt == 1:
                    #     sent = ' as the most inspiring work in the ' + branch_name + ' field, ' + sent
                    # elif cnt == 2:
                    #     sent = ' besides this one, several works contribute to the birth of [0]. ' + sent
                    sent = sent_upper(sent) # 大写
                    print(pid, sent)
                    sent_list.append(sent)
                    pid_list.append(pid)
                    title_list.append('['+str(cnt_glob)+'] '+title)
                    refer_text = get_papers_formated_reference([pid])[pid]
                    # results['tracing']['reference'][branch_name].append((pid, '[' + str(cnt_glob) + '] ' + title, score))
                    results['tracing']['reference'][branch_name].append((pid, '[' + str(cnt_glob+1) + '] ' + refer_text, score))
                    id_list.append(cnt_glob)
                    cnt += 1
                    cnt_glob += 1
                else:
                    print('no sent')
            else:
                pass
                print('no abstract')
        mx_weight = max(mx_weight, score_topic)
        topic_weights.append(score_topic)
        rel = pid2related_work(pid_list, sent_list, id_list)
        results['tracing']['survey'][branch_name] = rel
        text_list.append(rel)
        ref_list += title_list
        # results[branch_name] = (rel, title_list)
        print(branch_name, rel, title_list)
    for i in range(len(topic_weights)):
        topic_weights[i] /= mx_weight
    results['tracing']['topic_weights'] = topic_weights
    text = '\n'.join(text_list)
    return (text, ref_list)
def pro_branch_evol(branch_dic, rule=True):
    global results
    name = branch_dic['clusterNames']
    results['evolving']['topics'] = name
    branches = branch_dic['branches']
    cnt_glob = 0
    text_list, ref_list = [], []
    topic_weights = []
    mx_weight = 0
    for i, branch in enumerate(branches):
        pid2score = {}
        for dic in branch:
            pid2score[dic['paper_id']] = dic['score']
        branch_name = name[i]
        results['evolving']['reference'][branch_name] = []
        sent_list, id_list = [], []
        pid_list = []
        title_list = []
        score_topic = 0
        print(pid2score)
        sorted_list = sorted(pid2score.items(), key=lambda x: x[1], reverse=True)
        cnt = 1
        for i, item in enumerate(sorted_list):
            pid, score = item[0], item[1]
            score_topic += score
            abs = pid2abs(pid)
            if abs != '' and len(pid_list)<5:
                if rule:
                    sent = abs2sentence(abs, 100)
                else:
                    sent = get_sentence_from_abstract(abs)
                    sent = process_sentence(sent, 100)

                if sent != '':

                    title = pid2title(pid)
                    # if cnt == 1:
                    #     sent = ' as the work most inspired by [0] in the ' + branch_name + ' field, ' + sent
                    # elif cnt == 2:
                    #     sent = ' besides this one, several works were inspired by the idea of [0]. ' + sent
                    sent = sent_upper(sent)  # 大写
                    print(pid, sent)
                    sent_list.append(sent)
                    pid_list.append(pid)
                    title_list.append('['+str(cnt_glob)+'] '+title)
                    refer_text = get_papers_formated_reference([pid])[pid]
                    # results['evolving']['reference'][branch_name].append((pid, '['+str(cnt_glob)+'] '+title, score))
                    results['evolving']['reference'][branch_name].append((pid, '[' + str(cnt_glob+1) + '] ' + refer_text, score))
                    id_list.append(cnt_glob)
                    cnt += 1
                    cnt_glob += 1
                else:
                    print('no sent')
            else:
                pass
                print('no abstract')
        mx_weight = max(mx_weight, score_topic)
        topic_weights.append(score_topic)
        rel = pid2related_work(pid_list, sent_list, id_list)
        results['evolving']['survey'][branch_name] = rel
        text_list.append(rel)
        ref_list += title_list
        # results[branch_name] = (rel, title_list)
        print(branch_name, rel, title_list)
    for i in range(len(topic_weights)):
        topic_weights[i] /= mx_weight
    results['evolving']['topic_weights'] = topic_weights
    text = '\n'.join(text_list)
    return (text, ref_list)
def pro_root(root_pid, topics):
    abs = pid2abs(root_pid)
    sent = abs2sentence(abs, 0)
    start = 'This paper was influenced by a series of classical works in several fields, including '
    for i, topic in enumerate(topics):
        if i!=len(topics)-1:
            start += topic + ', '
        else:
            start += 'and ' + topic + '. '
    return sent + start
def pro_root_evol(root_pid, topics):
    abs = pid2abs(root_pid)
    sent = abs2sentence(abs, 0)
    start = 'After this work, a series of classical works in several fields, including '
    for i, topic in enumerate(topics):
        if i!=len(topics)-1:
            start += topic + ', '
        else:
            start += 'and ' + topic + ', '
    start += 'were influenced by it. '
    return sent + start
def pro_root_merge(root_pid, topics, topics_evol):
    abs = pid2abs(root_pid)
    sent = abs2sentence(abs, -1)
    if sent != "":
        sent = sent_upper(sent)  # 大写
        sent = past_tense(sent)
    start = ' This paper was influenced by a series of classical works in several fields, including '
    for i, topic in enumerate(topics):
        if i != len(topics) - 1:
            start += topic + ', '
        else:
            start += 'and ' + topic + '. '
    end = 'After this work, a series of classical works in several fields, including '
    for i, topic in enumerate(topics_evol):
        if i != len(topics_evol) - 1:
            end += topic + ', '
        else:
            end += 'and ' + topic + ', '
    end += 'were influenced by it. '
    return sent + start + end
def normalize1(name):
    return name.capitalize()
if __name__ == '__main__':

    rule = True
    branch_path = '../tracing_evolution_tree_jsons/'
    root_pids = ['77890282', '102038075', '151582811', '344930844', '491037503', '90534941', '261129775', '287664630', '425378987', '450884969']
    root_pids = ['3005757', '113768329', '169004034', '197806803', '247796706', '252362185', '372720438', '445475439']
    root_pids = ['287664630']
    db_write = pymysql.connect(host='xxxx', user='xxxx', passwd='xxxx',
                               db='xxxx', port=3306,
                               use_unicode=True, charset='utf8')
    cursor_write = db_write.cursor()
    for root_pid in root_pids:
        results = {'tracing': {'topics': [], 'reference': {}, 'survey': {}},
                   'evolving': {'topics': [], 'reference': {}, 'survey': {}}}
        '''
        file = os.listdir(branch_path)
        for f in file:
            file_path = path.join(branch_path, f)
        '''
        file_path = '../tracing_evolution_tree_jsons/ordered_txt_gen_'+root_pid+'_tracing.json'
        evol_file_path = '../tracing_evolution_tree_jsons/ordered_txt_gen_'+root_pid+'_evolution.json'
        # file_path = 'tree/output/'+root_pid + '_tracing.json'
        # evol_file_path = 'tree/output/'+root_pid + '_evolution.json'
        branch_dic = load_file(file_path)
        # 首字母大写
        for i in range(len(branch_dic['clusterNames'])):
            words = branch_dic['clusterNames'][i].split()
            words = list(map(normalize1, words))
            branch_dic['clusterNames'][i] = ' '.join(words)
        print(branch_dic)
        # root_text = pro_root(root_pid, branch_dic['clusterNames'])
        # results['tracing']['survey']['target_paper_survey'] = root_text
        # print(root_text)
        text, ref_list = pro_branch(branch_dic, rule)
        # text = root_text+'\n'+text
        print(text)
        for ref in ref_list:
            print(ref)

        evol_branch_dic = load_file(evol_file_path)
        # 首字母大写
        for i in range(len(evol_branch_dic['clusterNames'])):
            words = evol_branch_dic['clusterNames'][i].split()
            words = list(map(normalize1, words))
            evol_branch_dic['clusterNames'][i] = ' '.join(words)
        # root_text = pro_root_evol(root_pid, evol_branch_dic['clusterNames'])
        # results['evolving']['survey']['target_paper_survey'] = root_text
        root_text = pro_root_merge(root_pid, branch_dic['clusterNames'], evol_branch_dic['clusterNames'])
        print(root_text)
        results['target_paper_survey'] = root_text
        results['target_paper_title'] = pid2title(root_pid)
        # print(root_text)
        text, ref_list = pro_branch_evol(evol_branch_dic, rule)
        # text = root_text + '\n' + text
        print(text)
        for ref in ref_list:
            print(ref)
        print(results)
        with open('../machine_reading_jsons/result_'+root_pid+'.json', 'w', encoding='utf8')as f:
            json_str = json.dumps(results)
            f.write(json_str)


        with open('../machine_reading_jsons/result_'+root_pid+'.json', 'r', encoding='utf8') as f:
            result_json = json.load(f)
            result_json = json.dumps(result_json)