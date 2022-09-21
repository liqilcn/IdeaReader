from .data_provider import DataProvider
from .base import Publication, MrtType
from .utils import build_graph
from typing import Optional, Tuple, List, Iterable
from . import logger
import networkx as nx
import itertools

def retrieve(provider: DataProvider, query_pub: Publication, mrtType: MrtType, expected_size: int = 100) -> Tuple[List[Publication], nx.Graph]:
    pid2pub, max_depth = {query_pub.id: query_pub}, 0

    if mrtType == MrtType.UP:
        explore_ids = query_pub.references
        while len(pid2pub) < expected_size and len(explore_ids) > 0:
            max_depth += 1
            refs = provider.get_bulk(list(explore_ids))
            explore_ids = set()
            for ref in refs:
                ref.depth = max_depth
                pid2pub[ref.id] = ref
                explore_ids.update(ref.references)
            explore_ids.difference_update(pid2pub.keys())
            logger.debug('add %d references for depth %d' % (len(refs), max_depth))
    else:
        explore_ids = query_pub.citations
        while len(pid2pub) < expected_size and len(explore_ids) > 0:
            max_depth += 1
            cits = provider.get_bulk(list(explore_ids))
            explore_ids = set()
            for cit in cits:
                cit.depth = max_depth
                pid2pub[cit.id] = cit
                explore_ids.update(cit.citations)
            explore_ids.difference_update(pid2pub.keys())
            logger.debug('add %d citations for depth %d' % (len(cits), max_depth))

    logger.info('%d candidates retrieved' % len(pid2pub))

    # ensure all edges bidirectional
    for pid, pub in pid2pub.items():
        for rid in pub.references:
            if rid in pid2pub:
                pid2pub[rid].citations.add(pub.id)
        for cid in pub.citations:
            if cid in pid2pub:
                pid2pub[cid].references.add(pub.id)

    # build graph and calculate pagerank
    graph = build_graph(pid2pub=pid2pub)
    pagerank_scores = nx.algorithms.pagerank(graph.to_undirected())
    for pid, pagerank_score in pagerank_scores.items():
        pid2pub[pid].pagerank_score = pagerank_score

    # sort
    pubs = list(sorted(pid2pub.values(), key=lambda pub: (pub.depth, -pub.pagerank_score)))[:expected_size]
    return pubs