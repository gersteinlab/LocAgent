from datasets import load_dataset
import os
import logging
import json
import pandas as pd
from plugins.location_tools.utils.get_repo_structure.get_repo_structure import (
    get_project_structure_from_scratch,
)
from plugins.location_tools.utils.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
    get_full_file_paths_and_classes_and_functions,
    show_project_structure,
    line_wrap_content,
    transfer_arb_locs_to_locs,
    get_repo_files,
)
from plugins.location_tools.utils.repo import setup_github_repo
from copy import deepcopy
from typing import Optional
import fnmatch
from collections import defaultdict

# SET THIS IF YOU WANT TO USE THE PREPROCESSED FILES
PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC")
DEPENDENCY_GRAPH_LOC = os.environ.get("DEPENDENCY_GRAPH_LOC")
INDEX_STORE_LOC = os.environ.get("INDEX_STORE_LOC")
BM25_PERSIS_LOC = os.environ.get("BM25_PERSIS_LOC", 'index_data/SWE-bench_Lite/20241113_bm25_persist')
assert PROJECT_FILE_LOC != ''
assert DEPENDENCY_GRAPH_LOC != ''
assert INDEX_STORE_LOC != ''

def find_matching_files_from_list(file_list, file_pattern):
    """
    Find and return a list of file paths from the given list that match the given keyword or pattern.
    
    :param file_list: A list of file paths to search through.
    :param file_pattern: A keyword or pattern for file matching. Can be a simple keyword or a glob-style pattern.
    :return: A list of matching file paths
    """
    # If the pattern contains any of these glob-like characters, treat it as a glob pattern.
    if '*' in file_pattern or '?' in file_pattern or '[' in file_pattern:
        matching_files = fnmatch.filter(file_list, file_pattern)
    else:
        # Otherwise, treat it as a keyword search
        matching_files = [file for file in file_list if file_pattern in file]
    
    return matching_files

def get_meta_data(target_id, dataset:str="princeton-nlp/SWE-bench_Lite", split:str = "test"):
    # swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    swe_bench_data = load_dataset(dataset, split=split)
    bench_data = [x for x in swe_bench_data if x["instance_id"] == target_id][0]
    return bench_data

def get_repo_structures(bench_data):
    if PROJECT_FILE_LOC is not None:
        project_file = os.path.join(PROJECT_FILE_LOC, bench_data["instance_id"] + ".json")
        if os.path.exists(project_file):
            d = load_json(project_file)
        else:
            d = get_project_structure_from_scratch(
                bench_data["repo"], bench_data["base_commit"], bench_data["instance_id"], "playground"
            )
            if d:
                with open(project_file, 'w') as f:
                    json.dump(d, f, indent=4)
            else:
                # TODO: try catch
                return None
    else:
        logging.info("`PROJECT_FILE_LOC` is None, get the project structure from scratch")
        # we need to get the project structure directly
        d = get_project_structure_from_scratch(
            bench_data["repo"], bench_data["base_commit"], bench_data["instance_id"], "playground"
        )

    instance_id = d["instance_id"]
    structure = d["structure"]
    filter_none_python(structure)

    # some basic filtering steps
    # filter out test files (unless its pytest)
    if not d["instance_id"].startswith("pytest"):
        filter_out_test_files(structure)

    return structure

def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def write_jsonl(data, filepath):
    """
    Write data to a JSONL file at the given filepath.

    Arguments:
    data -- a list of dictionaries to write to the JSONL file
    filepath -- the path to the JSONL file to write
    """
    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def load_json(filepath):
    return json.load(open(filepath, "r"))


def combine_by_instance_id(data):
    """
    Combine data entries by their instance ID.

    Arguments:
    data -- a list of dictionaries with instance IDs and other information

    Returns:
    A list of combined dictionaries by instance ID with all associated data.
    """
    combined_data = defaultdict(lambda: defaultdict(list))
    for item in data:
        instance_id = item.get("instance_id")
        if not instance_id:
            continue
        for key, value in item.items():
            if key != "instance_id":
                combined_data[instance_id][key].extend(
                    value if isinstance(value, list) else [value]
                )
    return [
        {**{"instance_id": iid}, **details} for iid, details in combined_data.items()
    ]


# construct_topn_file_context
def construct_topn_file_context(
    file_to_locs,
    pred_files,
    file_contents,
    structure,
    context_window: int,
    loc_interval: bool = True,
    fine_grain_loc_only: bool = False,
    add_space: bool = False,
    sticky_scroll: bool = False,
    no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    file_loc_intervals = dict()
    topn_content = ""

    for pred_file, locs in file_to_locs.items():
        content = file_contents[pred_file]
        line_locs, context_intervals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )

        if len(line_locs) > 0:
            # Note that if no location is predicted, we exclude this file.
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals

def retrieve_graph(code_graph, graph_tags, search_term, structure, max_tags=100):
    one_hop_tags = []
    tags = []
    for tag in graph_tags:
        if tag['name'] == search_term and tag['kind'] == 'ref':
            tags.append(tag)
        if len(tags) >= max_tags:
            break
    # for tag in tags:
    for i, tag in enumerate(tags):
        # if i % 3 == 0:
        logging.info(f"Retrieving graph for {i}/{len(tags)}")
        # find corresponding calling function/class
        path = tag['rel_fname'].split('/')
        s = deepcopy(structure)   # stuck here
        for p in path:
            s = s[p]
        for txt in s['functions']:
            if tag['line'] >= txt['start_line'] and tag['line'] <= txt['end_line']:
                one_hop_tags.append((txt, tag['rel_fname']))  
        for txt in s['classes']:
            for func in txt['methods']:
                if tag['line'] >= func['start_line'] and tag['line'] <= func['end_line']:
                    func['text'].insert(0, txt['text'][0])
                    one_hop_tags.append((func, tag['rel_fname'])) 
    return one_hop_tags

def load_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test"
):
    data = load_dataset(dataset_name, split=split)
    return {d["instance_id"]: d for d in data}

def load_instance(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
):
    data = load_instances(dataset_name, split=split)
    return data[instance_id]

def setup_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = f"swe-bench/{repo_dir_name}"
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )

def setup_full_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    # repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = instance_data["repo"]
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )

def get_repo_dir_name(repo: str):
    return repo.replace("/", "_")

import re
def is_test(name, test_phrases=None):
    if test_phrases is None:
        test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", name.lower()))
    return any(word in words for word in test_phrases)


def is_legal_variable_name(name):
    # Regex pattern for a valid Python identifier (variable name)
    valid_variable_pattern = re.compile(r'^[_a-zA-Z][_a-zA-Z0-9]*$')
    return valid_variable_pattern.match(name)


def extract_module_id(text: str):
    # Regular expression pattern to extract the file_name, ntype, and module_name
    pattern = r'(?P<file_name>[\w/]+\.py)\s\((?P<ntype>\w+):\s(?P<module_name>[\w.]+)\)'
    
    # Use re.search to match the pattern
    match = re.search(pattern, text)

    if match:
        file_name = match.group('file_name')
        module_name = match.group('module_name')
        ntype = match.group('ntype')
        module_id = f'{file_name}:{module_name}'
        return (module_id, ntype)
    else:
        return (None, None)
    
    
def get_formatted_node_str(nid, nodes_data):
    if ':' in nid:
        file_name, module_name = nid.split(':')
        for node in nodes_data:
            if node['file_path'] == file_name and node['module_name'] == module_name:
                break
        ntype = node['type']
        formatted_text = f'{file_name} ({ntype}: {module_name})'
    else:
        formatted_text = nid

    return formatted_text