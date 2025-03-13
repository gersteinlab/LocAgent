from rapidfuzz import process, fuzz
import re
import pickle
from typing import Dict, List, Optional
import networkx as nx
from graph_encoder.dependency_graph.traverse_graph_v2 import is_test_file
from graph_encoder.dependency_graph.build_graph_v2 import (
    NODE_TYPE_DIRECTORY,
    NODE_TYPE_FILE,
    NODE_TYPE_CLASS,
    NODE_TYPE_FUNCTION,
)

NTYPES = [
    NODE_TYPE_DIRECTORY,
    NODE_TYPE_FILE,
    NODE_TYPE_FUNCTION,
    NODE_TYPE_CLASS,
]


def fuzzy_retrieve_from_graph_nodes(
    keyword: str,
    graph_path : Optional[str] = None,
    graph: Optional[nx.MultiDiGraph] = None,
    search_scope: str = 'all', # enum = {'function', 'class', 'file', 'all'}
    include_files: Optional[str] = None,
    similarity_top_k: int = 5,
    return_score: bool = False,
):
    assert graph_path or isinstance(graph, nx.MultiDiGraph)
    assert search_scope in NTYPES or search_scope == 'all'

    if graph_path:
        graph = pickle.load(open(graph_path, "rb"))

    selected_nids = list()
    filter_nids = list()
    for nid in graph:
        if is_test_file(nid): continue
        ndata = graph.nodes[nid]
        if search_scope == 'all' and ndata['type'] in NTYPES[1:]:
            nfile = nid.split(':')[0]
            if not include_files or nfile in include_files:
                filter_nids.append(nid)
            selected_nids.append(nid)
        elif ndata['type'] == search_scope:
            nfile = nid.split(':')[0]
            if not include_files or nfile in include_files:
                filter_nids.append(nid)
            selected_nids.append(nid)
    
    if not filter_nids:
        filter_nids = selected_nids
        
    # Custom function to split tokens on underscores and hyphens
    def custom_tokenizer(s):
        return re.findall(r'\b\w+\b', s.replace('_', ' ').replace('-', ' '))

    # Use token_set_ratio with custom tokenizer
    matches = process.extract(
        keyword,
        filter_nids,
        scorer=fuzz.token_set_ratio,
        processor=lambda s: ' '.join(custom_tokenizer(s)),
        limit=similarity_top_k
    )
    if not return_score:
        return_nids = [match[0] for match in matches]
        return return_nids
    
    # matches: List[Tuple(nid, score)]
    return matches