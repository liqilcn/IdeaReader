import json
import os

import sys
sys.path.append("..")

from mrtframework import MasterReadingTree


def get_pub_dict_for_text_generation(compact_pub_dict) -> dict:
    return {"paper_id": compact_pub_dict["paper_id"], "score": compact_pub_dict["score"]}


def get_ordered_dict_for_text_generation(compact_dict: dict) -> dict:
    # 根据 importance 对 branches 和 clusterNames 重排序
    compact_dict["branches"] = [x for _, x in sorted(zip(compact_dict["importance"], compact_dict["branches"]), reverse=True)]
    compact_dict["clusterNames"] = [x for _, x in sorted(zip(compact_dict["importance"], compact_dict["clusterNames"]), reverse=True)]
    compact_dict["tagGroups"] = [x for _, x in sorted(zip(compact_dict["importance"], compact_dict["tagGroups"]), reverse=True)]
    list.sort(compact_dict["importance"], reverse=True)

    ans_dict = {}
    branches_list = []
    for cluster in compact_dict["branches"]:
        cluster_list = []
        for timeline in cluster:
            for pub in timeline:
                cluster_list.append(get_pub_dict_for_text_generation(pub))
        branches_list.append(cluster_list)
    ans_dict["branches"] = branches_list
    ans_dict["clusterNames"] = compact_dict["clusterNames"]
    return ans_dict


def mrt_to_json_for_text_generation(tree: MasterReadingTree, output_dir: str, suffix: str) -> bool:
    compact_tree = tree.to_compact_json()
    tree_for_text_generation = get_ordered_dict_for_text_generation(compact_tree)
    if output_dir is None:
        return False
    with open(os.path.join(output_dir, '{pid}_{suffix}.json'.format(pid=tree.pubs[0].id, suffix=suffix)), "w") as outfile:
        json.dump(tree_for_text_generation, outfile)
    return True

