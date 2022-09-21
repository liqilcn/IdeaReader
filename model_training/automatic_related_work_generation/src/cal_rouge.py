import os
import time
import json
from multiprocessing import Pool

import shutil
import codecs

from others import pyrouge

class Rouge_Cal_Args():
    def __init__(self):
        self.is_json = False
        self.target_model = 'bertsumabs'# 'bertsumabs' or 'bart'
        self.preprocess_dataset = 'delve'
        self.candidate_file = f'../results/{self.target_model}_result/{self.preprocess_dataset}/{115000}.candidate'
        self.gold_file = f'../results/{self.target_model}_result/{self.preprocess_dataset}/{115000}.gold'
        self.process_num = 4

def process(data):
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = "rouge-tmp-{}-{}".format(current_time,pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        # 删掉所有临时文件
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def test_rouge(cand, ref, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    assert len(candidates) == len(references)
    candidates_chunks = list(chunks(candidates, int(len(candidates)/num_processes))) # 将candidates列表和references列表分段，分到不同进程处理
    references_chunks = list(chunks(references, int(len(references)/num_processes)))
    n_pool = len(candidates_chunks)
    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i],references_chunks[i],i))
    pool = Pool(n_pool)
    results = pool.map(process,arg_lst)
    final_results = {}
    for i,r in enumerate(results):
        for k in r:
            if(k not in final_results):
                final_results[k] = r[k]*len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k]/len(candidates)
    return final_results
def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
    results_dict["rouge_1_recall"] * 100,
    results_dict["rouge_2_recall"] * 100,
    results_dict["rouge_l_recall"] * 100

    # ,results_dict["rouge_su*_f_score"] * 100
    )


if __name__ == "__main__":
    # 总体思路就是将candidates和reference拆成一一对应的cand和ref对，每个cand和每个ref都存储为一个txt文件，用rouge155计算每个cand和ref对
    # 之间的ROUGE-1, ROUGE-2, ROUGE-L （都是取Recall和F1 score），以ROUGE-1的Recall为例，相当于计算每个cand和ref对的ROUGE-1的Recall
    # 再求平均，这样计算方式也易于并行
    args = Rouge_Cal_Args()
    print(args.candidate_file)
    print(args.gold_file)
    print(args.process_num)
    if args.is_json:
        candidates = json.load(open(f'{args.candidate_file}.json','r'))
        references = json.load(open(f'{args.gold_file}.json','r'))
    else:
        candidates = codecs.open(args.candidate_file, encoding="utf-8")  # 这个函数读取能避免繁琐的编码不统一的问题，与open功能相同
        references = codecs.open(args.gold_file, encoding="utf-8")

    results_dict = test_rouge(candidates, references, args.process_num)
    # return 0
    print(time.strftime('%H:%M:%S', time.localtime())
)
    print(rouge_results_to_str(results_dict))
    # logger.info(rouge_results_to_str(results_dict))