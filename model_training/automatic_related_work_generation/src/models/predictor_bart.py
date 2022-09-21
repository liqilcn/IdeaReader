import os
import json
import codecs
import torch
from tqdm import tqdm

def bart_generator(args, test_iter, model, tokenizer, step):
    # 生成生成文本文件，ref文件
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    with torch.no_grad():
        final_candidates = []
        final_references = []
        for batch in tqdm(test_iter):
            src, ref = batch
            inputs = tokenizer(list(src), max_length=args.src_max_length, truncation=True, padding=True, return_tensors="pt")  # 不同字符串长度不同，需要padding
            inputs.to(device)
            summary_ids = model.generate(inputs["input_ids"], num_beams=args.beam_size, min_length=args.min_length, max_length=args.max_length)
            summary_list = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            final_candidates += summary_list
            for r in list(ref):
                final_references.append(r)
        gold_path = os.path.join(args.result_path, f'{step}.gold')
        can_path = os.path.join(args.result_path, f'{step}.candidate')
        gold_json_path = os.path.join(args.result_path, f'{step}.gold.json')
        can_json_path = os.path.join(args.result_path, f'{step}.candidate.json')
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        can_out_file = codecs.open(can_path, 'w', 'utf-8')
        print(len(final_candidates))
        print(len(final_references))
        json.dump(final_candidates, open(can_json_path, 'w'))
        json.dump(final_references, open(gold_json_path, 'w'))
        for c in final_candidates:
            can_out_file.write(c + '\n')
        for r in final_references:
            gold_out_file.write(r + '\n')

