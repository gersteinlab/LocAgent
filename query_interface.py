import json
import os
import queue
import time
from pathlib import Path
from typing import Dict, Optional

import networkx as nx

from dependency_graph.build_graph import build_graph, write_graph_outputs, build_bm25_index
from dependency_graph.traverse_graph import RepoEntitySearcher
from plugins.location_tools.repo_ops.repo_ops import set_current_repo, reset_current_repo
from plugins import LocationToolsRequirement
from util.config import load_config
from util.prompts.prompt import PromptManager
from util.prompts.pipelines import auto_search_prompt as auto_search
from util.runtime import function_calling
from util.process_output import get_loc_results_from_raw_outputs, merge_sample_locations
import auto_search_main

GRAPH_FILE = 'graph.pkl'
TAGS_FILE = 'tags.jsonl'
BM25_DIR = 'bm25'
META_FILE = 'index_meta.json'


def index_repo(repo_path: str, output_dir: str) -> Dict[str, str]:
    repo_path = os.path.abspath(repo_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    graph = build_graph(repo_path, global_import=True)
    graph_path = os.path.join(output_dir, GRAPH_FILE)
    tags_path = os.path.join(output_dir, TAGS_FILE)
    write_graph_outputs(graph, graph_path, tags_path)

    bm25_dir = os.path.join(output_dir, BM25_DIR)
    build_bm25_index(repo_path, bm25_dir)

    meta = {
        'repo_path': repo_path,
        'graph_path': graph_path,
        'tags_path': tags_path,
        'bm25_dir': bm25_dir,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    meta_path = os.path.join(output_dir, META_FILE)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return meta


def _load_meta(index_dir: str) -> Dict[str, str]:
    meta_path = os.path.join(index_dir, META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _build_query_instruction(query: str, repo_name: str) -> str:
    instruction = auto_search.TASK_INSTRUECTION.format(package_name=repo_name)
    instruction += f"\nProblem statement:\n{query}\n"
    return instruction


def _run_agent(query: str, repo_name: str, config: Dict[str, object]):
    prompt_manager = PromptManager(
        prompt_dir=os.path.join(os.path.dirname(__file__), 'util/prompts'),
        agent_skills_docs=LocationToolsRequirement.documentation,
    )

    system_prompt = function_calling.SYSTEM_PROMPT if config.get('use_function_calling', True) else prompt_manager.system_message
    messages = [{
        'role': 'system',
        'content': system_prompt,
    }]

    messages.append({
        'role': 'user',
        'content': _build_query_instruction(query, repo_name),
    })

    tools = None
    if config.get('use_function_calling', True):
        tools = function_calling.get_tools(
            codeact_enable_search_keyword=True,
            codeact_enable_search_entity=True,
            codeact_enable_tree_structure_traverser=True,
            simple_desc=config.get('simple_desc', False),
        )

    result_queue: queue.SimpleQueue = queue.SimpleQueue()
    auto_search_main.auto_search_process(
        result_queue=result_queue,
        model_name=str(config.get('model')),
        messages=messages,
        fake_user_msg=auto_search.FAKE_USER_MSG_FOR_LOC,
        temp=float(config.get('temperature', 0.7)),
        tools=tools,
        use_function_calling=config.get('use_function_calling', True),
        max_iteration_num=int(config.get('max_iterations', 20)),
    )

    result = result_queue.get()
    if isinstance(result, dict) and result.get('type') == 'BadRequestError':
        raise RuntimeError(result.get('error'))

    final_output, _messages, traj_data = result
    return final_output, traj_data


def _rank_locations(graph: nx.MultiDiGraph, raw_outputs: list[str]):
    found_files, found_modules, found_entities = get_loc_results_from_raw_outputs(
        instance_id=None,
        raw_outputs=raw_outputs,
        graph=graph,
    )
    ranked_files, ranked_modules, ranked_entities = merge_sample_locations(
        found_files, found_modules, found_entities, ranking_method='mrr'
    )
    return ranked_files, ranked_modules, ranked_entities


def _build_dependency_snapshot(graph: nx.MultiDiGraph, touched: set[str]):
    edges = []
    for u, v, data in graph.edges(data=True):
        if u in touched and v in touched:
            edges.append({'from': u, 'to': v, 'type': data.get('type')})
    return edges


def query_repo(index_dir: str, query: str, output_path: Optional[str] = None, config_path: Optional[str] = None):
    config, _ = load_config(config_path)
    meta = _load_meta(index_dir)

    graph_path = meta.get('graph_path', os.path.join(index_dir, GRAPH_FILE))
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f'Graph index not found at {graph_path}')

    repo_path = meta.get('repo_path', index_dir)
    repo_name = os.path.basename(repo_path.rstrip('/')) or 'repository'

    set_current_repo(repo_path=repo_path, graph_index_path=graph_path)

    try:
        final_output, traj_data = _run_agent(query, repo_name, config)

        import pickle
        graph = pickle.load(open(graph_path, 'rb'))
        ranked_files, ranked_modules, ranked_entities = _rank_locations(graph, [final_output])

        searcher = RepoEntitySearcher(graph)
        localized_entities = []
        for entity in ranked_entities:
            data = searcher.get_node_data([entity])[0]
            localized_entities.append({
                'file': entity.split(':', 1)[0],
                'entity': entity.split(':', 1)[1] if ':' in entity else entity,
                'type': data.get('type'),
                'start_line': data.get('start_line', 1),
                'end_line': data.get('end_line', 1),
                'relevance': 'Agent-localized entity.',
            })

        touched = set(ranked_files + ranked_modules + ranked_entities)
        result = {
            'query': query,
            'localized_files': ranked_files,
            'localized_entities': localized_entities,
            'dependency_graph': {
                'touched_components': list(touched),
                'edges': _build_dependency_snapshot(graph, touched),
            },
            'agent_reasoning': final_output,
            'usage': traj_data.get('usage', {}),
        }

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

        return result
    finally:
        reset_current_repo()


def interactive(index_dir: str, config_path: Optional[str] = None):
    config, _ = load_config(config_path)
    meta = _load_meta(index_dir)
    graph_path = meta.get('graph_path', os.path.join(index_dir, GRAPH_FILE))
    repo_path = meta.get('repo_path', index_dir)
    repo_name = os.path.basename(repo_path.rstrip('/')) or 'repository'

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f'Graph index not found at {graph_path}')

    set_current_repo(repo_path=repo_path, graph_index_path=graph_path)
    try:
        while True:
            query = input('query> ').strip()
            if not query:
                continue
            if query.lower() in {'exit', 'quit'}:
                break

            final_output, traj_data = _run_agent(query, repo_name, config)
            import pickle
            graph = pickle.load(open(graph_path, 'rb'))
            ranked_files, ranked_modules, ranked_entities = _rank_locations(graph, [final_output])
            print('\n'.join(ranked_files[:10]))
    finally:
        reset_current_repo()
