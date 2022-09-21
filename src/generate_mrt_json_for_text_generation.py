import argparse
import concurrent
import json
import logging
from typing import Optional, List

from data_utils import acemap_utils, json_utils
from mrtframework import MasterReadingTree
from mrtframework.base import MrtType
from mrtframework.data_provider import DataProvider

# 执行MRT的文章所需的最少引用数或被引数
MIN_RC_COUNT = 10


def storeTracingTrees(pids: List[str], run_parallel: bool, device: str, output_dir: Optional[str] = None, workers: int = 16) -> None:
    pids = list(set(pids))  # 去重
    valid_pids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for i, cnt in enumerate(list(executor.map(acemap_utils.getReferenceCount, pids))):
            if cnt >= MIN_RC_COUNT:
                valid_pids.append(pids[i])

    provider = DataProvider(downloader=acemap_utils.acemap_unsort_downloader)

    def MasterReadingTreeWrapper(pid: str) -> None:
        try:
            tree = MasterReadingTree(provider=provider, query_pub=provider.get(pid), mrtType=MrtType.UP, device=device)
            json_utils.mrt_to_json_for_text_generation(tree, output_dir, "tracing")
        except Exception as e:
            print("Error when generating tracing tree of pid = " + str(pid))
            print(e)
        return

    if run_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(MasterReadingTreeWrapper, valid_pids)
    else:
        for pid in valid_pids:
            MasterReadingTreeWrapper(pid)

    return


def storeEvolutionTrees(pids: List[str], run_parallel: bool, device: str, output_dir: Optional[str] = None, workers: int = 16) -> None:
    pids = list(set(pids))  # 去重
    valid_pids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for i, cnt in enumerate(list(executor.map(acemap_utils.getCitationCount, pids))):
            if cnt >= MIN_RC_COUNT:
                valid_pids.append(pids[i])

    provider = DataProvider(downloader=acemap_utils.acemap_sort_downloader)

    def MasterReadingTreeWrapper(pid: str) -> None:
        try:
            tree = MasterReadingTree(provider=provider, query_pub=provider.get(pid), mrtType=MrtType.DOWN, device=device)
            json_utils.mrt_to_json_for_text_generation(tree, output_dir, "evolution")
        except Exception as e:
            print("Error when generating evolution tree of pid = " + str(pid))
            print(e)
        return

    if run_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(MasterReadingTreeWrapper, valid_pids)
    else:
        for pid in valid_pids:
            MasterReadingTreeWrapper(pid)

    return


if __name__ == '__main__':
    # json格式说明
    # pub_ids：paper id的列表
    # mrt_type：可选3种："tracing"（溯源树）、"evolution"（脉络树）、"both"（两种都生成）
    # verbosity：可选"INFO"和"WARNING"等
    # run_parallel: 为0时单线程串行执行，为1时多线程
    # workers: 线程数（仅在run_parallel为1时有效）
    # device: cpu/cuda/cuda:1等等

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="./input.json",
                        help='the path of input JSON file')
    parser.add_argument('--output_dir', type=str, default="../tracing_evolution_tree_jsons/",
                        help='the output directory for generated results')

    args = parser.parse_args()

    f = open(args.input_path)
    param = json.load(f)
    f.close()

    logging.root.setLevel(param["verbosity"])
    logging.info("log starts")

    if param["mrt_type"] == "tracing" or param["mrt_type"] == "both":
        storeTracingTrees(param["pub_ids"], param["run_parallel"] == 1, param["device"], args.output_dir, workers=param["workers"])
    if param["mrt_type"] == "evolution" or param["mrt_type"] == "both":
        storeEvolutionTrees(param["pub_ids"], param["run_parallel"] == 1, param["device"], args.output_dir, workers=param["workers"])
