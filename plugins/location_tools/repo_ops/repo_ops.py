"""repo_ops.py

This module provides various file manipulation skills for the agent.

Functions:
- get_repo_structure(): Get the structure of the repository where the issue resides.
- get_directory_structure(dir_path: str):Search and display the directory structure of a given repository, starting from a specified directory path.
- get_file_content(file_path: str): Get the content of the given file.
- get_file_structures(file_list: list[str]): Get the skeleton/structure of the given files, which you can get all the classes and functions in these files.
- search_class(class_name: str, file_pattern: Optional[str] = None): Searching for the specified class name within the codebase and retrieves its implementation. 
- search_class_structures(class_names: list[str], file_pattern: Optional[str] = None): Retrieves class definitions (skeletons) from the codebase based on the provided class names, filtered by an optional file pattern.
- search_method(method_name: str, class_name: Optional[str] = None, file_pattern: Optional[str] = None): Searches for specified method name within the codebase and retrieves its definitions along with relevant context.
- search_invoke_and_reference(module_name: str, file_path: str, ntype: str): Analyzes the dependencies of a specific class, or function, identifying how they interact with other parts of the codebase. This is useful for understanding the broader impact of any changes or issues and ensuring that all related components are accounted for.
- search_interactions_among_modules(module_ids: list[str]): Analyze the interactions between specified modules by examining a pre-built dependency graph.
- search_in_repo(search_terms: list[str], file_pattern: Optional[str] = "**/*.py"): Performs a combined search using both the BM25 algorithm and semantic search on the codebase.
"""

import pickle
import json
import os
import re
from collections import defaultdict
from typing import List, Optional, Union, Tuple, Dict
import collections
from copy import deepcopy
import uuid
# from xmlrpc.client import boolean
from click import argument
import networkx as nx
from graph_encoder.dependency_graph import RepoSearcher, RepoEntitySearcher
from graph_encoder.dependency_graph.build_graph_v2 import (
    NODE_TYPE_DIRECTORY, NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION,
    EDGE_TYPE_CONTAINS, EDGE_TYPE_INHERITS, EDGE_TYPE_INVOKES, EDGE_TYPE_IMPORTS, VALID_NODE_TYPES,
    VALID_EDGE_TYPES
)
from graph_encoder.dependency_graph.traverse_graph_v2 import is_test_file, traverse_tree_structure, \
    traverse_graph_structure, RepoDependencySearcher, traverse_json_structure
from plugins.location_tools.utils.util import (
    get_meta_data,
    get_repo_structures,
    construct_topn_file_context,
    retrieve_graph,
    setup_swebench_repo, setup_full_swebench_repo,
    get_repo_dir_name,
    find_matching_files_from_list,
    DEPENDENCY_GRAPH_LOC,
    INDEX_STORE_LOC, BM25_PERSIS_LOC,
    # REPO_GRAPH_LOC, 
    is_test,
    is_legal_variable_name,
    get_formatted_node_str,
    extract_module_id,

)
from plugins.location_tools.utils.compress_file import get_skeleton
from plugins.location_tools.utils.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    show_project_structure,
    get_repo_files,
    line_wrap_content,
    merge_intervals,
)
from plugins.location_tools.retriever.bm25_retriever import (
    build_code_retriever_from_repo as build_code_retriever,
    build_module_retriever_from_graph as build_module_retriever,
    build_retriever_from_persist_dir as load_retriever,
)
from plugins.location_tools.retriever.fuzzy_retriever import (
    fuzzy_retrieve_from_graph_nodes as fuzzy_retrieve
)
from plugins.location_tools.utils.result_format import (
    QueryInfo,
    QueryResult,
)

import Stemmer
from repo_index.workspace import Workspace
from llama_index.retrievers.bm25 import BM25Retriever

import logging

logger = logging.getLogger(__name__)

CURRENT_ISSUE_ID: str | None = None
CURRENT_INSTANCE: dict | None = None
CURRENT_STRUCTURE: dict | None = None
ALL_FILE: list | None = None
ALL_CLASS: list | None = None
ALL_FUNC: list | None = None
REPO_SAVE_DIR: str | None = None
FOUND_MODULES: list[str] = []

DP_GRAPH_ENTITY_SEARCHER: RepoEntitySearcher | None = None
DP_GRAPH_DEPENDENCY_SEARCHER: RepoDependencySearcher | None = None
DP_GRAPH: nx.MultiDiGraph | None = None


def add_found_modules(file_path: str, module_name: str = None, ntype: str = 'file'):
    global FOUND_MODULES
    if ntype == 'file':
        module_id = f'{file_path}'
    else:
        module_id = f'{file_path}:{module_name}'
    if module_id not in FOUND_MODULES:
        FOUND_MODULES.append(module_id)


def get_found_modules():
    global FOUND_MODULES
    return FOUND_MODULES


def set_current_issue(instance_id: str = None, instance_data: dict = None,
                      dataset: str = "princeton-nlp/SWE-bench_Lite", split: str = "test", rank=0):
    global CURRENT_ISSUE_ID, CURRENT_INSTANCE, CURRENT_STRUCTURE
    global ALL_FILE, ALL_CLASS, ALL_FUNC
    assert instance_id or instance_data

    if instance_id:
        CURRENT_ISSUE_ID = instance_id
        CURRENT_INSTANCE = get_meta_data(CURRENT_ISSUE_ID, dataset, split)
    else:
        CURRENT_ISSUE_ID = instance_data['instance_id']
        CURRENT_INSTANCE = instance_data

    CURRENT_STRUCTURE = get_repo_structures(CURRENT_INSTANCE)
    ALL_FILE, ALL_CLASS, ALL_FUNC = get_full_file_paths_and_classes_and_functions(CURRENT_STRUCTURE)

    global REPO_SAVE_DIR
    # Generate a temperary folder and add uuid to avoid collision
    REPO_SAVE_DIR = os.path.join('playground', str(uuid.uuid4()))
    # assert playground doesn't exist
    assert not os.path.exists(REPO_SAVE_DIR), f"{REPO_SAVE_DIR} already exists"
    # create playground
    os.makedirs(REPO_SAVE_DIR)

    # setup graph traverser
    global DP_GRAPH_ENTITY_SEARCHER, DP_GRAPH_DEPENDENCY_SEARCHER, DP_GRAPH
    G = pickle.load(
        open(f"{DEPENDENCY_GRAPH_LOC}/{CURRENT_ISSUE_ID}.pkl", "rb")
    )
    DP_GRAPH_ENTITY_SEARCHER = RepoEntitySearcher(G)
    DP_GRAPH_DEPENDENCY_SEARCHER = RepoDependencySearcher(G)
    DP_GRAPH = G

    logging.debug(f'Rank = {rank}, set CURRENT_ISSUE_ID = {CURRENT_ISSUE_ID}')


import subprocess
def reset_current_issue():
    global CURRENT_ISSUE_ID, CURRENT_INSTANCE, CURRENT_STRUCTURE, FOUND_MODULES
    CURRENT_ISSUE_ID = None
    CURRENT_INSTANCE = None
    CURRENT_STRUCTURE = None
    FOUND_MODULES = []

    global ALL_FILE, ALL_CLASS, ALL_FUNC
    ALL_FILE, ALL_CLASS, ALL_FUNC = None, None, None

    global REPO_SAVE_DIR
    subprocess.run(
        ["rm", "-rf", REPO_SAVE_DIR], check=True
    )
    REPO_SAVE_DIR = None


def get_current_issue_id():
    global CURRENT_ISSUE_ID
    return CURRENT_ISSUE_ID


def get_current_issue_structure():
    global CURRENT_STRUCTURE
    return CURRENT_STRUCTURE


def get_current_repo_modules():
    global ALL_FILE, ALL_CLASS, ALL_FUNC
    return ALL_FILE, ALL_CLASS, ALL_FUNC


def get_current_issue_data():
    global CURRENT_ISSUE_ID, CURRENT_INSTANCE, CURRENT_STRUCTURE
    return CURRENT_ISSUE_ID, CURRENT_INSTANCE, CURRENT_STRUCTURE


def get_repo_save_dir():
    global REPO_SAVE_DIR
    return REPO_SAVE_DIR


def get_graph_entity_searcher() -> RepoEntitySearcher:
    global DP_GRAPH_ENTITY_SEARCHER
    return DP_GRAPH_ENTITY_SEARCHER


def get_graph_dependency_searcher() -> RepoDependencySearcher:
    global DP_GRAPH_DEPENDENCY_SEARCHER
    return DP_GRAPH_DEPENDENCY_SEARCHER


def get_graph():
    global DP_GRAPH
    assert DP_GRAPH is not None
    return DP_GRAPH


file_content_in_block_template = """
file: {file_name}
```
{content}
```
"""


def get_repo_structure() -> str:
    """Get the structure of the repository where the issue resides, which you can use to understand the repo and then search related files.

    Args:
        None
    
    Returns:
        str: The tree structure of the repository where the issue resides.
    """

    structure = get_current_issue_structure()

    # only structure
    structure_str = show_project_structure(structure).strip()
    return structure_str


# sub-structure
def get_directory_structure(dir_path: str) -> str:
    """Search and display the directory structure of a given repository, starting from a specified directory path.

    Args:
        dir_path (str): Directory path to search, formatted as a forward-slash (/) separated string.
    Returns:
        str: The structure of the directory specified in `dir_path`.
    """
    structure = get_current_issue_structure()
    message, current_dir = "", ""

    def show_directory_str(dn, spacing=0):
        return " " * spacing + str(dn) + "/" + "\n"

    path = dir_path.split('/')
    path = [p.strip() for p in path if p.strip()]
    if not path:
        structure_str = show_project_structure(structure).strip()
        message += f"Invalid directory path '{dir_path}'.\n"
        message += f"Show the structure of the whole repository.\n"
        return message + structure_str

    s = deepcopy(structure)
    for i, p in enumerate(path):
        if p in s:
            # spacing = i*4
            current_dir = '/'.join(path[:i + 1]) + '/' + '\n'
            s = s[p]
        else:
            if i == 0:
                message += f"No directory named '{p}' in the root.\n"
                message += f"Show the structure of the whole repository.\n"
            else:
                message += f"No directory named '{p}' under '{current_dir}'.\n"
                message += f"Show the structure under '{current_dir}'.\n"
            break

    if i == 0:
        sub_structure_str = show_project_structure(s).strip()
    else:
        sub_structure_str = show_project_structure(s, spacing=4)

    structure_str = message + current_dir + sub_structure_str
    return structure_str


def is_valid_file(file_name: str):
    files, _, _ = get_current_repo_modules()

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    valid_file_paths = list(set(all_file_paths) - set(exclude_files))
    if file_name in valid_file_paths:
        return True
    else:
        return False


def get_file_content_(file_name: str, return_str=False):
    files, _, _ = get_current_repo_modules()

    for file_content in files:
        if file_content[0] == file_name:
            if return_str:
                content = "\n".join(file_content[1])
                return content
            else:
                return file_content[1]
    return None


def get_file_content(file_path: str) -> str:
    """Get the entire content of the given file.

    Args:
        file_path: str: The selected file (path) which might be related to this issue.
    
    Returns:
        str: A string containing the entire content of the corresponding file.
    
    """

    if not is_valid_file(file_path):
        return "Invalid file path."

    file_content = get_file_content_(file_path, return_str=True)
    search_result_str = file_content_in_block_template.format(file_name=file_path, content=file_content)
    return search_result_str


def get_file_structures(file_list: list[str]) -> str:
    """Get the skeleton/structure of the given files, which you can get all the classes and functions in these files.

    Args:
        file_list: list[str]: The selected files (paths) which might be related to this issue, and the length of this array is not more than 5.
    
    Returns:
        str: The skeleton/structure of the given files.
    """
    if isinstance(file_list, str):
        file_list = [file_list]
    if not isinstance(file_list, list):
        return None

    search_results = []
    for file_name in file_list:
        if not is_valid_file(file_name):
            search_results.append(f"Invalid file path: {file_name}.")
            continue

        file_content = get_file_content_(file_name, return_str=True)
        file_skeleton = get_skeleton(file_content)
        content = file_content_in_block_template.format(file_name=file_name, content=file_skeleton)
        search_results.append(content)

    search_result_str = "\n".join(search_results)
    return search_result_str


def construct_file_contexts(file: str, found_related_locs: list[str]) -> str:
    """This function is used to get specific lines or chunks of code within a Python file that are related to a particular issue, based on the provided file path and related classes or functions within this file.

    Args:
        file: str: The file path within the repository that are suspected to contain code relevant to the issue.
        found_related_locs: list[str]: The classes or functions defined within `file`, e.g., `["class: class_A","function: class_A.func_1", "function: func_2"]`.
    
    Returns:
        str: The specific lines or chunks of code within given file.
    """
    model_found_files = [file]
    if not isinstance(model_found_files, list):
        return None
    found_related_locs = [found_related_locs]
    issue_id, bench_data, structure = get_current_issue_data()
    file_contents = get_repo_files(structure, model_found_files)

    coarse_found_locs = {}
    for i, pred_file in enumerate(model_found_files):
        if len(found_related_locs) > i:
            coarse_found_locs[pred_file] = found_related_locs[i]

    topn_content, file_loc_intervals = construct_topn_file_context(
        coarse_found_locs,
        model_found_files,
        file_contents,
        structure,
        context_window=10,
        loc_interval=True,
        add_space=False,
        sticky_scroll=False,
        no_line_number=False,
    )
    return topn_content


def search_class_structures(class_names: list[str], file_pattern: Optional[str] = None) -> str:
    """Retrieves class definitions (skeletons) from the codebase based on the provided class names, filtered by an optional file pattern.

    Args:
        class_names (list[str]): List of class names to search for.
        file_pattern (Optional[str]): A glob pattern to filter the files to search in. Defaults to None, meaning all files are searched.

    Returns:
        str: A formatted string containing file paths, class names, and the method signatures for each class.
    """
    class_content_in_block_template = """
Found class `{cls_name}` in file `{file_name}`:
```
{content}
```
"""
    if isinstance(class_names, str):
        class_names = re.split(r'[\s,]+', class_names)

    # issue_id, bench_data, structure = get_current_issue_data()
    files, classes, _ = get_current_repo_modules()

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    include_files = all_file_paths
    if file_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_pattern)
    if not include_files:
        file_pattern = None
        include_files = all_file_paths

    all_search_results = []
    for class_name in class_names:
        matched_cls = [cls for cls in classes if class_name == cls['name']]
        if not matched_cls:
            matched_cls = [cls for cls in classes if class_name in cls['name'] or cls['name'] in class_name]

        filter_matched_cls = filter_class(matched_cls, include_files, exclude_files)
        if file_pattern and not filter_matched_cls:
            filter_matched_cls = filter_class(matched_cls, all_file_paths, exclude_files)

        search_results = []
        for cls in filter_matched_cls:
            file = cls['file']
            # class_contents[file] = dict()
            content = get_file_content_(cls['file'])
            # class_contents[file][cls['name']] = "\n".join(content[cls['start_line']-1 : cls['end_line']])
            class_content = "\n".join(content[cls['start_line'] - 1: cls['end_line']])
            search_result = class_content_in_block_template.format(
                cls_name=cls['name'],
                file_name=cls['file'],
                content=get_skeleton(class_content)
            )
            search_results.append(search_result)

        if search_results:
            all_search_results.append("\n".join(search_results))
        else:
            all_search_results.append(f"Found no results for class `{class_name}`.")

    class_strucures = "\n".join(all_search_results)
    return class_strucures


def search_class_contents(class_name: str, exact_match: bool = False) -> dict:
    files, classes, _ = get_current_repo_modules()

    matched_cls = [cls for cls in classes if class_name == cls['name']]
    if not exact_match and not matched_cls:
        matched_cls = [cls for cls in classes if class_name in cls['name'] or cls['name'] in class_name]

    class_contents = dict()
    for cls in matched_cls:
        file = cls['file']
        class_contents[file] = dict()
        content = get_file_content_(cls['file'])
        class_contents[file][cls['name']] = "\n".join(content[cls['start_line'] - 1: cls['end_line']])

    return class_contents


def filter_class(class_data: list, include_files: list, exclude_files: list):
    filtered_cls = []
    for cls in class_data:
        file = cls['file']
        if file not in exclude_files and file in include_files:
            filtered_cls.append(cls)
    return filtered_cls


def search_class(class_name: str, file_pattern: Optional[str] = None) -> str:
    """Searching for specified class name within the codebase and retrieves their implementation.
    This function is essential for quickly locating class implementations, understanding their structures, and analyzing how they fit into the overall architecture of the project.

    Args:
        class_names: str: The class name to search for in the codebase. Please search one class name at a time.
        file_pattern: Optional[str]: A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.

    Returns:
        str: A formatted string containing the search results for each specified class, including code snippets of their definitions.
    """

    files, classes, _ = get_current_repo_modules()

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    include_files = all_file_paths
    if file_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_pattern)
    if not include_files:
        file_pattern = None
        include_files = all_file_paths

    matched_cls = [cls for cls in classes if class_name == cls['name']]
    if not matched_cls:
        matched_cls = [cls for cls in classes if class_name in cls['name'] or cls['name'] in class_name]

    filter_matched_cls = filter_class(matched_cls, include_files, exclude_files)
    if file_pattern and not filter_matched_cls:
        filter_matched_cls = filter_class(matched_cls, all_file_paths, exclude_files)

    all_search_results = []
    # class_contents = dict()
    for cls in filter_matched_cls:
        content = get_file_content_(cls['file'], return_str=True)
        if not content:
            continue
        cls_content = line_wrap_content(content, [(cls['start_line'], cls['end_line'])])
        file_name, cls_name = cls['file'], cls['name']

        search_result = f'Found class `{cls_name}` in `{file_name}`:\n'
        search_result += f'\n{cls_content}\n\n'

        # add dependencies
        dependencies_data = search_invoke_and_reference(cls_name, file_name, ntype='class')
        if dependencies_data:
            search_result += dependencies_data
        all_search_results.append(search_result)

    class_contents = '\n'.join(all_search_results)
    return class_contents


def search_function(func_name: str,
                    all_functions: list[dict],
                    exact_match=True,
                    include_files: list[str] = None):
    matched_funcs = [func for func in all_functions if func['name'] == func_name]
    if not exact_match and not matched_funcs:
        matched_funcs = [func for func in all_functions if
                         func['name'] in func_name or func_name in func['name']]

    filtered_matched_funcs = []
    for func in matched_funcs:
        file = func['file']
        if include_files and file in include_files:
            filtered_matched_funcs.append(func)
    return filtered_matched_funcs


def get_method_content(method, class_name, file_name, file_content: str):
    search_result = 'Found method `{class_name}.{method_name}` in file `{file_name}`:\n'.format(
        class_name=class_name,
        method_name=method['name'],
        file_name=file_name
    )
    method_content = line_wrap_content(file_content, [(method['start_line'], method['end_line'])])
    search_result += f'\n{method_content}\n'
    return search_result


def search_method(method_name: str, class_name: Optional[str] = None,
                  file_pattern: Optional[str] = None) -> str:
    """Search for specified method name within the codebase and retrieves its definitions along with relevant context. If a class name is provided, the search is limited to methods within that class. This function is essential for quickly locating method implementations, understanding their behaviors, and analyzing their roles in the codebase.
    
    Args:
        method_name (str): The method (function) name to search for in the codebase.
        class_name (Optional[str]): The name of the class to limit the search scope to methods within this class. If None, the search includes methods in all classes and global functions.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
    
    Returns:
        str: A formatted string containing the search result for the specified method, including messages and code snippets of its definition and context.
    """
    # issue_id, bench_data, structure = get_current_issue_data()
    files, classes, functions = get_current_repo_modules()

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    all_valid_file_paths = list(set(all_file_paths) - set(exclude_files))
    if file_pattern:
        include_files = find_matching_files_from_list(all_valid_file_paths, file_pattern)
        if not include_files:
            file_pattern = None
            include_files = all_valid_file_paths
    else:
        include_files = all_valid_file_paths

    init_method_name = method_name
    if '.' in method_name:
        method_name = method_name.split('.')[-1]

    search_results = []
    found_method = False

    if class_name:
        add_dependency = False
        # Exact Match first
        cls_names = re.split(r'\s+|\s*\.\s*', class_name)
        cls_names = [item for item in cls_names if item and is_legal_variable_name(item)]

        matched_cls = [cls for cls in classes if cls['name'] in cls_names and cls['file'] in include_files]
        if not matched_cls and file_pattern:
            # ignore file_pattern
            matched_cls = [cls for cls in classes if
                           cls['name'] in cls_names and cls['file'] in all_valid_file_paths]

        if matched_cls:
            add_dependency = True
        else:
            search_result = f'Found no class named `{class_name}`.'
            search_results.append(search_result)
            # fuzzy search for class
            matched_cls = [cls for cls in classes if
                           (cls['name'] in class_name or class_name in cls['name']) and cls[
                               'file'] in include_files]

        if not matched_cls and file_pattern:
            matched_cls = [cls for cls in classes if
                           (cls['name'] in class_name or class_name in cls['name']) and cls[
                               'file'] in all_valid_file_paths]

        for cls in matched_cls:
            matched_methods = [method for method in cls['methods'] if method['name'] == method_name]
            if matched_methods:
                found_method = True
                file_content = get_file_content_(cls['file'], return_str=True)
            for method in matched_methods:
                search_result = get_method_content(method, cls['name'], cls['file'], file_content)
                method_sig = '{class_name}.{method_name}'.format(
                    class_name=cls['name'],
                    method_name=method['name']
                )

                if add_dependency:
                    # add dependencies
                    dependencies_data = search_invoke_and_reference(method_sig, cls['file'], ntype='function')
                    if dependencies_data:
                        search_result += dependencies_data

                search_results.append(search_result)

        # search any class which has such a method
        if not found_method:
            for cls in classes:
                matched_methods = [method for method in cls['methods'] if method['name'] == method_name]
                if not matched_methods:
                    continue
                if matched_methods:
                    found_method = True
                    file_content = get_file_content_(cls['file'], return_str=True)
                for method in matched_methods:
                    search_result = get_method_content(method, cls['name'], cls['file'], file_content)
                    search_results.append(search_result)

        if not found_method:
            search_result = f"Found no results for method `{init_method_name}` in any class."
            search_results.append(search_result)

    if not found_method:
        # search function directly
        add_dependency = False
        # func_names = [func for func in method_name.split() if is_legal_variable_name(func)]

        # Exact match first
        filtered_matched_funcs = search_function(method_name, functions, exclude_files, include_files)
        if not filtered_matched_funcs and file_pattern:
            filtered_matched_funcs = search_function(method_name, functions, all_valid_file_paths)

        if filtered_matched_funcs:
            add_dependency = True
        else:
            # fuzzy search
            filtered_matched_funcs = search_function(method_name, functions, False, include_files)
        if not filtered_matched_funcs and file_pattern:
            filtered_matched_funcs = search_function(method_name, functions, exact_match=False,
                                                     include_files=all_valid_file_paths)

        for func in filtered_matched_funcs:
            found_method = True
            content = get_file_content_(func['file'], return_str=True)
            if not content:
                continue
            search_result = 'Found method `{method_name}` in file `{file}`:\n'.format(
                method_name=func['name'],
                file=func['file']
            )
            func_content = line_wrap_content(content, [(func['start_line'], func['end_line'])])
            search_result += f'\n{func_content}\n\n'

            if add_dependency:
                # add dependencies
                dependencies_data = search_invoke_and_reference(func['name'], func['file'], ntype='function')
                if dependencies_data:
                    search_result += dependencies_data
                search_results.append(search_result)

    search_result_str = ""
    if search_results:
        search_result_str = "\n".join(search_results)

    if not found_method:
        search_result_str += f"Found no results for method {init_method_name}.\n"
        if class_name:
            search_result_str += f"Searching for class `{class_name}` ...\n"
            search_result_str += search_class_structures([class_name], file_pattern)

    return search_result_str


def get_module_name_by_line_num(file_path: str, line_num: int):
    # TODO: 如果line不属于类内某个函数且该类长度较大，那么找到临近两个成员函数并返回
    
    entity_searcher = get_graph_entity_searcher()
    dp_searcher = get_graph_dependency_searcher()

    cur_module = None
    if entity_searcher.has_node(file_path):
        module_nids, _ = dp_searcher.get_neighbors(file_path, etype_filter=[EDGE_TYPE_CONTAINS])
        module_ndatas = entity_searcher.get_node_data(module_nids)
        for module in module_ndatas:
            if module['start_line'] <= line_num <= module['end_line']:
                cur_module = module  # ['node_id']
                break
        if cur_module and cur_module['type'] == NODE_TYPE_CLASS:
            func_nids, _ = dp_searcher.get_neighbors(cur_module['node_id'], etype_filter=[EDGE_TYPE_CONTAINS])
            func_ndatas = entity_searcher.get_node_data(func_nids, return_code_content=True)
            for func in func_ndatas:
                if func['start_line'] <= line_num <= func['end_line']:
                    cur_module = func  # ['node_id']
                    break

    if cur_module: # and cur_module['type'] in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]
        return cur_module
        # module_ndata = entity_searcher.get_node_data([cur_module['node_id']], return_code_content=True)
        # return module_ndata[0]
    return None


def get_code_block_by_line_nums(query_info, context_window=20):
    # file_path: str, line_nums: List[int]
    searcher = get_graph_entity_searcher()
    
    file_path = query_info.file_path_or_pattern
    line_nums = query_info.line_nums
    cur_query_results = []
    
    # file_content = get_file_content_(file_path, return_str=False)
    file_data = searcher.get_node_data([file_path], return_code_content=False)[0]
    line_intervals = []
    res_modules = []
    # res_code_blocks = None
    for line in line_nums:
        # 首先检查是哪个module的代码
        module_data = get_module_name_by_line_num(file_path, line)
        
        # 如果不是某个module, 则搜索上下20行
        if not module_data:
            min_line_num = max(1, line - context_window)
            max_line_num = min(file_data['end_line'], line + context_window)
            line_intervals.append((min_line_num, max_line_num))
            
        elif module_data['node_id'] not in res_modules:
            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                       nid=module_data['node_id'],
                                       ntype=module_data['type'],
                                       start_line=module_data['start_line'],
                                       end_line=module_data['end_line'],
                                       retrieve_src=f"Retrieved code context including {query_info.term}."
                                       )
            cur_query_results.append(query_result)
            res_modules.append(module_data['node_id'])
            
    if line_intervals:
        line_intervals = merge_intervals(line_intervals)
        for interval in line_intervals:
            start_line, end_line = interval
            query_result = QueryResult(query_info=query_info, 
                                        format_mode='code_snippet',
                                        nid=file_path,
                                        file_path=file_path,
                                        start_line=start_line,
                                        end_line=end_line,
                                        retrieve_src=f"Retrieved code context including {query_info.term}."
                                        )
            cur_query_results.append(query_result)
        # res_code_blocks = line_wrap_content('\n'.join(file_content), line_intervals)

    # return res_code_blocks, res_modules
    return cur_query_results


def search_in_repo(search_terms: list[str], file_pattern: Optional[str] = "**/*.py") -> str:
    """Performs a combined search using both the BM25 algorithm and semantic search on the codebase. 
    For each search term, this function retrieves code snippets by first performing a BM25 search to rank documents based on 
    the similarity to the term and then follows up with a semantic search to find more contextually similar code snippets.
    
    Args:
        search_terms (list[str]): Textual queries used to search the codebase. Each can be a functional description, a potential class or method name, or any relevant terms related to the code you want to find in the repository.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
    
    Returns:
        str: A formatted string containing the combined results from both BM25 and semantic search. 
            Each search result contains the file path and the retrieved code snippet (the partial code of a module or just the skeleton of the specific module).
    """
    # issue_id, instance, structure = get_current_issue_data()
    files, classes, functions = get_current_repo_modules()
    all_file_paths = [file[0] for file in files]
    all_class_names = [cls['name'] for cls in classes]
    all_function_names = [func['name'] for func in functions]

    result = ""
    for term in search_terms:
        result += f"Searching for '{term}' ...\n\n## Searching Result:\n"
        cur_result = ''
        if term in all_class_names:
            cur_result += search_class_structures([term], file_pattern)
        if term in all_function_names:
            cur_result += search_method(term, file_pattern)
            if 'Found no results' in cur_result:
                cur_result = ''
        if '.' in term and len(term.split('.')) == 2:
            cur_result += search_method(term.split('.')[-1], term.split('.')[0], file_pattern)
            if 'Found no results' in cur_result:
                cur_result = ''

        if cur_result:
            result += cur_result
        else:
            result += search_term_in_repo(term, file_pattern)
    return result


def parse_node_id(nid: str):
    nfile = nid.split(':')[0]
    nname = nid.split(':')[-1]
    return nfile, nname


def search_entity_in_global_dict(term: str, include_files: Optional[List[str]] = None, prefix_term=None):
    searcher = get_graph_entity_searcher()
    
    # TODO: hard code cases like "class Migration" and "function testing"
    if term.startswith(('class ', 'Class')):
        term = term[len('class '):].strip()
    elif term.startswith(('function ', 'Function ')):
        term = term[len('function '):].strip()
    elif term.startswith(('method ', 'Method ')):
        term = term[len('method '):].strip()
    elif term.startswith('def '):
        term = term[len('def '):].strip()
    
    # TODO: lower case if not find
    # TODO: filename xxx.py as key (also lowercase if not find)
    # global_name_dict = None
    if term in searcher.global_name_dict:
        global_name_dict = searcher.global_name_dict
        nids = global_name_dict[term]
    elif term.lower() in searcher.global_name_dict_lowercase:
        term = term.lower()
        global_name_dict = searcher.global_name_dict_lowercase
        nids = global_name_dict[term]
    else:
        return None
    
    node_datas = searcher.get_node_data(nids, return_code_content=False)
    found_entities_filter_dict = collections.defaultdict(list)
    for ndata in node_datas:
        nfile, _ = parse_node_id(ndata['node_id'])
        if not include_files or nfile in include_files:
            prefix_terms = []
            # candidite_prefixes = ndata['node_id'].lower().replace('.py', '').replace('/', '.').split('.')
            candidite_prefixes = re.split(r'[./:]', ndata['node_id'].lower().replace('.py', ''))[:-1]
            if prefix_term:
                prefix_terms = prefix_term.lower().split('.')
            if not prefix_term or all([prefix in candidite_prefixes for prefix in prefix_terms]):
                found_entities_filter_dict[ndata['type']].append(ndata['node_id'])

    return found_entities_filter_dict


def search_entity(query_info, include_files: List[str] = None):
    term = query_info.term
    searcher = get_graph_entity_searcher()
    # cur_result = ''
    continue_search = True

    cur_query_results = []
    
    # first: exact match in graph
    if searcher.has_node(term):
        continue_search = False
        query_result = QueryResult(query_info=query_info, format_mode='complete', nid=term,
                                   retrieve_src=f"Exact match found for entity name `{term}`."
                                   )
        cur_query_results.append(query_result)
    
    # TODO: __init__ not exsit
    elif term.endswith('.__init__'):
        nid = term[:-(len('.__init__'))]
        if searcher.has_node(nid):
            continue_search = False
            node_data = searcher.get_node_data([nid], return_code_content=True)[0]
            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                    nid=nid, 
                                    ntype=node_data['type'],
                                    start_line=node_data['start_line'],
                                    end_line=node_data['end_line'],
                                    retrieve_src=f"Exact match found for entity name `{nid}`."
                                    )
            cur_query_results.append(query_result)
    
    # second: search in global name dict
    if continue_search: 
        found_entities_dict = search_entity_in_global_dict(term, include_files)
        if not found_entities_dict:
            found_entities_dict = search_entity_in_global_dict(term)
        
        use_sub_term = False
        used_term = term
        if not found_entities_dict and '.' in term:
            # for cases: class_name.method_name
            try:
                prefix_term = '.'.join(term.split('.')[:-1]).split()[-1] # incase of 'class '/ 'function '
            except IndexError:
                prefix_term = None
            split_term = term.split('.')[-1].strip()
            used_term = split_term
            found_entities_dict = search_entity_in_global_dict(split_term, include_files, prefix_term)
            if not found_entities_dict:
                found_entities_dict = search_entity_in_global_dict(split_term, prefix_term)
            if not found_entities_dict:
                use_sub_term = True
                found_entities_dict = search_entity_in_global_dict(split_term)
        
        # TODO: split the term and find in global dict
            
        if found_entities_dict:
            for ntype, nids in found_entities_dict.items():
                if not nids: continue
                # if not continue_search: break

                # class 和 function 逻辑一致(3个以内显示)
                if ntype in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS, NODE_TYPE_FILE]:
                    if len(nids) <= 3:
                        node_datas = searcher.get_node_data(nids, return_code_content=True)
                        for ndata in node_datas:
                            query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                                       nid=ndata['node_id'], 
                                                       ntype=ndata['type'],
                                                       start_line=ndata['start_line'],
                                                       end_line=ndata['end_line'],
                                                       retrieve_src=f"Match found for entity name `{used_term}`."
                                                       )
                            cur_query_results.append(query_result)
                        # continue_search = False
                    else:
                        node_datas = searcher.get_node_data(nids, return_code_content=False)
                        for ndata in node_datas:
                            query_result = QueryResult(query_info=query_info, format_mode='fold', 
                                                       nid=ndata['node_id'],
                                                       ntype=ndata['type'],
                                                       retrieve_src=f"Match found for entity name `{used_term}`."
                                                       )
                            cur_query_results.append(query_result)
                        if not use_sub_term:
                            continue_search = False
                        else:
                            continue_search = True
                                   
        
    # third: bm25 search (entity + content)
    if continue_search:
        module_nids = []

        # # 搜索时加不加file?
        # # if not any(symbol in file_path_or_pattern for symbol in ['*','?', '[', ']']):
        # term_with_file = f'{file_path_or_pattern}:{term}'
        # module_nids = bm25_module_retrieve(query=term_with_file, include_files=include_files)

        # 根据keyword搜索module
        module_nids = bm25_module_retrieve(query=term, include_files=include_files)
        if not module_nids:
            module_nids = bm25_module_retrieve(query=term)
            
        if not module_nids:
            # result += f"No entity found using BM25 search. Try to use fuzzy search...\n"
            module_nids = fuzzy_retrieve(term, graph=get_graph(), similarity_top_k=3)

        module_datas = searcher.get_node_data(module_nids, return_code_content=True)
        showed_module_num = 0
        for module in module_datas[:5]:
            if module['type'] in [NODE_TYPE_FILE, NODE_TYPE_DIRECTORY]:
                query_result = QueryResult(query_info=query_info, format_mode='fold', 
                                        nid=module['node_id'],
                                        ntype=module['type'],
                                        retrieve_src=f"Retrieved entity using keyword search (bm25)."
                                        )
                cur_query_results.append(query_result)
            elif showed_module_num < 3:
                showed_module_num += 1
                query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                        nid=module['node_id'],
                                        ntype=module['type'],
                                        start_line=module['start_line'],
                                            end_line=module['end_line'],
                                            retrieve_src=f"Retrieved entity using keyword search (bm25)."
                                        )
                cur_query_results.append(query_result)

    return (cur_query_results, continue_search)


def merge_query_results(query_results):
    priority = ['complete', 'code_snippet', 'preview', 'fold']
    merged_results = {}
    all_query_results: List[QueryResult] = []

    for qr in query_results:
        if qr.format_mode == 'code_snippet':
            all_query_results.append(qr)
        
        elif qr.nid and qr.nid in merged_results:
            # Merge query_info_list
            if qr.query_info_list[0] not in merged_results[qr.nid].query_info_list:
                merged_results[qr.nid].query_info_list.extend(qr.query_info_list)

            # Select the format_mode with the highest priority
            existing_format_mode = merged_results[qr.nid].format_mode
            if priority.index(qr.format_mode) < priority.index(existing_format_mode):
                merged_results[qr.nid].format_mode = qr.format_mode
                merged_results[qr.nid].start_line = qr.start_line
                merged_results[qr.nid].end_line = qr.end_line
                merged_results[qr.nid].retrieve_src = qr.retrieve_src
                
        elif qr.nid:
            merged_results[qr.nid] = qr
    
    # print('found_code_sippets', len(found_code_sippets))
    # print('merged_node_results', len(list(merged_results.values())))
    all_query_results += list(merged_results.values())
    # print('merged_results', len(all_query_results))
    return all_query_results


def rank_and_aggr_query_results(query_results, fixed_query_info_list):
    query_info_list_dict = {}

    for qr in query_results:
        # Convert the query_info_list to a tuple so it can be used as a dictionary key
        key = tuple(qr.query_info_list)

        if key in query_info_list_dict:
            query_info_list_dict[key].append(qr)
        else:
            query_info_list_dict[key] = [qr]
            
    # for the key: sort by query
    def sorting_key(key):
        # Find the first matching element index from fixed_query_info_list in the key (tuple of query_info_list)
        for i, fixed_query in enumerate(fixed_query_info_list):
            if fixed_query in key:
                return i
        # If no match is found, assign a large index to push it to the end
        return len(fixed_query_info_list)

    sorted_keys = sorted(query_info_list_dict.keys(), key=sorting_key)
    sorted_query_info_list_dict = {key: query_info_list_dict[key] for key in sorted_keys}
    
    # for the value: sort by format priority
    priority = {'complete': 1, 'code_snippet': 2, 'preview': 3,  'fold': 4}  # Lower value indicates higher priority
    # TODO: merge the same node in 'code_snippet' and 'preview'
    
    organized_dict = {}
    for key, values in sorted_query_info_list_dict.items():
        nested_dict = {priority_key: [] for priority_key in priority.keys()}
        for qr in values:
            # Place the qr in the nested dictionary based on its format_mode
            if qr.format_mode in nested_dict:
                nested_dict[qr.format_mode].append(qr)

        # Only add keys with non-empty lists to keep the result clean
        organized_dict[key] = {k: v for k, v in nested_dict.items() if v}
    
    return organized_dict
        

def search_code_snippets(
        search_terms: Optional[List[str]] = None,
        line_nums: Optional[List] = None,
        file_path_or_pattern: Optional[str] = "**/*.py",
) -> str:
    """Searches the codebase to retrieve relevant code snippets based on given queries(terms or line numbers).
    
    This function supports retrieving the complete content of a code entity, 
    searching for code entities such as classes or functions by keywords, or locating specific lines within a file. 
    It also supports filtering searches based on a file path or file pattern.
    
    Note:
    1. If `search_terms` are provided, it searches for code snippets based on each term:
        - If a term is formatted as 'file_path:QualifiedName' (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum') ,
          or just 'file_path', the corresponding complete code is retrieved or file content is retrieved.
        - If a term matches a file, class, or function name, matched entities are retrieved.
        - If there is no match with any module name, it attempts to find code snippets that likely contain the term.
        
    2. If `line_nums` is provided, it searches for code snippets at the specified lines within the file defined by 
       `file_path_or_pattern`.

    Args:
        search_terms (Optional[List[str]]): A list of names, keywords, or code snippets to search for within the codebase. 
            Terms can be formatted as 'file_path:QualifiedName' to search for a specific module or entity within a file 
            (e.g., 'src/helpers/math_helpers.py:MathUtils.calculate_sum') or as 'file_path' to retrieve the complete content 
            of a file. This can also include potential function names, class names, or general code fragments.

        line_nums (Optional[List[int]]): Specific line numbers to locate code snippets within a specified file. 
            When provided, `file_path_or_pattern` must specify a valid file path.
        
        file_path_or_pattern (Optional[str]): A glob pattern or specific file path used to filter search results 
            to particular files or directories. Defaults to '**/*.py', meaning all Python files are searched by default.
            If `line_nums` are provided, this must specify a specific file path.

    Returns:
        str: The search results, which may include code snippets, matching entities, or complete file content.
        
    
    Example Usage:
        # Search for the full content of a specific file
        result = search_code_snippets(search_terms=['src/my_file.py'])
        
        # Search for a specific function
        result = search_code_snippets(search_terms=['src/my_file.py:MyClass.func_name'])
        
        # Search for specific lines (10 and 15) within a file
        result = search_code_snippets(line_nums=[10, 15], file_path_or_pattern='src/example.py')
        
        # Combined search for a module name and within a specific file pattern
        result = search_code_snippets(search_terms=["MyClass"], file_path_or_pattern="src/**/*.py")
    """
    
    # TODO: [by-gangda] (Functionality Requirements) What do we want to search?
    # - Keywords (w. restriction of file_path_or_pattern) -> Node IDs + Code Snippets
    # - Line Numbers (w. qualified file_path) -> Node IDs + Code Snippets, the file_path must be validated and completed
    # - Node ID -> Complete Code

    # TODO: [by-gangda] (Implementation) How can we perform the search?
    # 1. Get Node ID
    #     - if query term already a qualified Node ID -> Node ID
    #     - if query term a keyword
    #         1) is it a module name? -> use global_dict  TODO: Hard Coding for case 'class Migration', not include
    #         2) is it an incomplete file_path:module_name? -> use bm25 to search on all Node IDs
    #         3) is it a global variable or code snippet? -> use bm25 to search on content
    #       -> Node ID
    #     - if query term is a line number
    #         1) Is file_path valid? If not, ignore it and print warning.
    #         2) Locate the module and get its Node ID. If it belongs to global variables, save (File Node ID, line number).
    #            If the line number is out of range, ignore it and print warning.
    # 2. Get Code Snippets and Formatting
    #     - Deduplicate identified Node IDs.
    #     - Collect Special Parameters from the previous stage. For example, return complete code or preview? retrieved by bm25 or global_dict? Line number.
    #     - Separately handle different types of nodes.

    files, _, _ = get_current_repo_modules()
    all_file_paths = [file[0] for file in files]

    result = ""
    # exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    if file_path_or_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_path_or_pattern)
        if not include_files:
            include_files = all_file_paths
            result += f"No files found for file pattern '{file_path_or_pattern}'. Will search all files.\n...\n"
    else:
        include_files = all_file_paths

    query_info_list = []
    all_query_results = []
    
    if search_terms:
        # 所有搜索项一起搜
        filter_terms = []
        for term in search_terms:
            if is_test_file(term):
                result += f'No results for test files: `{term}`. Please do not search for any test files.\n\n'
            else:
                filter_terms.append(term)
        
        joint_terms = deepcopy(filter_terms)
        if len(filter_terms) > 1:
            filter_terms.append(' '.join(filter_terms))
        
        for i, term in enumerate(filter_terms):
            term = term.strip().strip('.')
            if not term: continue
                
            query_info = QueryInfo(term=term)
            query_info_list.append(query_info)
            
            cur_query_results = []
            
            # search entity
            query_results, continue_search = search_entity(query_info=query_info, include_files=include_files)
            cur_query_results.extend(query_results)
            
            # search content
            if continue_search:
                query_results = bm25_code_retrieve(query_info=query_info, include_files=include_files)
                cur_query_results.extend(query_results)
                
            elif i != (len(filter_terms)-1):
                joint_terms[i] = ''
                filter_terms[-1] = ' '.join([t for t in joint_terms if t.strip()])
                if filter_terms[-1] in filter_terms[:-1]:
                    filter_terms[-1] = ''
                
            all_query_results.extend(cur_query_results)
    
    if file_path_or_pattern in all_file_paths and line_nums:
        if isinstance(line_nums, int):
            line_nums = [line_nums]
        file_path = file_path_or_pattern
        term = file_path + ':line ' + ', '.join([str(line) for line in line_nums])
        # result += f"Search `line(s) {line_nums}` in file `{file_path}` ...\n"
        query_info = QueryInfo(term=term, line_nums=line_nums, file_path_or_pattern=file_path)
        
        # 根据文件名和行号搜索代码
        query_results = get_code_block_by_line_nums(query_info)
        all_query_results.extend(query_results)
    
    
    merged_results = merge_query_results(all_query_results)
    ranked_query_to_results = rank_and_aggr_query_results(merged_results, query_info_list)
    
    
    # format output
    # format_mode: 'complete', 'preview', 'code_snippet', 'fold': 4
    searcher = get_graph_entity_searcher()
    
    for query_infos, format_to_results in ranked_query_to_results.items():
        term_desc = ', '.join([f'"{query.term}"' for query in query_infos])
        result += f'##Searching for term {term_desc}...\n'
        result += f'### Search Result:\n'
        cur_result = ''
        for format_mode, query_results in format_to_results.items():
            if format_mode == 'fold':
                cur_retrieve_src = ''
                for qr in query_results:
                    if not cur_retrieve_src:
                        cur_retrieve_src = qr.retrieve_src
                        
                    if cur_retrieve_src != qr.retrieve_src:
                        cur_result += "Source: " + cur_retrieve_src + '\n\n'
                        cur_retrieve_src = qr.retrieve_src
                        
                    cur_result += qr.format_output(searcher)
                    
                cur_result += "Source: " + cur_retrieve_src + '\n'
                if len(query_results) > 1:
                    cur_result += 'Hint: Use more detailed query to get the full content of some if needed.\n'
                else:
                    cur_result += f'Hint: Search `{query_results[0].nid}` for the full content if needed.\n'
                cur_result += '\n'
                
            elif format_mode == 'complete':
                for qr in query_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'

            elif format_mode == 'preview':
                # 去掉小模块, 只留下大模块
                filtered_results = []
                grouped_by_file = defaultdict(list)
                for qr in query_results:
                    if (qr.end_line - qr.start_line) < 100:
                        grouped_by_file[qr.file_path].append(qr)
                    else:
                        filtered_results.append(qr)
                
                for file_path, results in grouped_by_file.items():
                    # Sort by start_line and then by end_line in descending order
                    sorted_results = sorted(results, key=lambda qr: (qr.start_line, -qr.end_line))

                    max_end_line = -1
                    for qr in sorted_results:
                        # If the current QueryResult's range is not completely covered by the largest range seen so far, keep it
                        if qr.end_line > max_end_line:
                            filtered_results.append(qr)
                            max_end_line = max(max_end_line, qr.end_line)
                
                # filtered_results = query_results
                for qr in filtered_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'
            
            elif format_mode == 'code_snippet':
                for qr in query_results:
                    cur_result += qr.format_output(searcher)
                    cur_result += '\n'
            
        cur_result += '\n\n'
        
        if cur_result.strip():
            result += cur_result
        else:
            result += 'No locations found.\n\n'
        
    return result.strip()


def get_entity_contents(entity_names: List[str]):
    searcher = get_graph_entity_searcher()
    
    result = ''
    for name in entity_names:
        name = name.strip().strip('.')
        if not name: continue
        
        result += f'##Searching for entity `{name}`...\n'
        result += f'### Search Result:\n'
        query_info = QueryInfo(term=name)
        
        if searcher.has_node(name):
            query_result = QueryResult(query_info=query_info, format_mode='complete', nid=name,
                                    retrieve_src=f"Exact match found for entity name `{name}`."
                                    )
            result += query_result.format_output(searcher)
            result += '\n\n'
        else:
            result += 'Invalid name. \nHint: Valid entity name should be formatted as "file_path:QualifiedName" or just "file_path".'
            result += '\n\n'
    return result.strip()


def search_repo_by_json_obj(search_terms: list):
    # searcher = get_graph_traverser()

    search_results = ""
    for term in search_terms:
        keyword = term['keyword']
        file_path = term['possible_file_path']
        possible_line_numbers = term['possible_line_numbers']
        if possible_line_numbers and file_path and not keyword:
            search_results += search_in_repo(search_mode='line', line_nums=possible_line_numbers,
                                             file_path_or_pattern=file_path)
            continue

        if keyword and not possible_line_numbers:
            # if file_path:
            #     keyword = f'{file_path}:{keyword}'
            search_results += search_in_repo(search_mode='keyword', search_terms=[keyword],
                                             file_path_or_pattern=file_path)
            continue

        # keyword / possible_line_numbers 同时存在
        searched_modules = []
        for line_num in possible_line_numbers:
            module = get_module_name_by_line_num(file_path, line_num)
            if module and module['node_id'] not in searched_modules:
                searched_modules.append(module['node_id'])
                if keyword in module['code_content'] or keyword in module['node_id']:
                    if module['type'] == NODE_TYPE_FUNCTION:
                        search_results += f"Searching for '{keyword}' and line numbers {possible_line_numbers} in `{file_path}`...\n"
                        code_content = module['code_content'].split('\n')
                        code_content = '\n'.join(code_content[1:-1])
                        func_name = module['node_id'].split(':')[-1]
                        search_results += '## Search Result:'
                        search_results += f'Found {NODE_TYPE_FUNCTION} `{func_name}` in file `{file_path}`.'
                        search_results += f'\n```\n...\n{code_content}\n...\n```\n\n'
                    else:
                        search_results += search_in_repo(search_mode='line', line_nums=possible_line_numbers,
                                                         file_path_or_pattern=file_path)
                    continue

        if searched_modules:
            search_results += search_in_repo(search_mode='line', line_nums=possible_line_numbers,
                                             file_path_or_pattern=file_path)
            search_results += '\n'

        # if file_path:
        #     keyword = f'{file_path}:{keyword}'
        search_results += search_in_repo(search_mode='keyword', search_terms=[keyword],
                                         file_path_or_pattern=file_path)
        search_results += '\n'

    return search_results


def code_retriever(search_terms: list[str], file_pattern: Optional[str] = "**/*.py", return_code=False):
    result = ""
    for query in search_terms:
        bm25_result = "### Retrieving by bm25 algorithm:\n"
        bm25_result += bm25_code_retrieve(query=query, file_pattern=file_pattern, similarity_top_k=10)

        semantic_result = "### Retrieving by semantic search:\n"
        semantic_result += semantic_search(query=query, file_pattern=file_pattern, max_results=10,
                                           use_skeleton=True)

        retrieve_result = bm25_result + '\n' + semantic_result
        result += retrieve_result
        return retrieve_result


def search_term_in_repo(query: str, file_pattern: Optional[str] = "**/*.py") -> str:
    """Performs a combined search using both the BM25 algorithm and semantic search on the codebase. 
    This function retrieves code snippets by first performing a BM25 search to rank documents based on 
    the similarity to the query and then follows up with a semantic search to find more contextually 
    similar code snippets.
    
    Args:
        query (str): A textual query used to search the codebase. This can be a functional description, a potential class or method name, or any relevant terms related to the code you want to find in the repository.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
    
    Returns:
        str: A formatted string containing the combined results from both BM25 and semantic search, including file paths and the retrieved code snippets (the partial code of a module or the skeleton of the specific module).
    """
    bm25_result = "### Retrieving by bm25 algorithm:\n"
    bm25_result += bm25_retrieve(query=query, file_pattern=file_pattern, similarity_top_k=10)

    semantic_result = "### Retrieving by semantic search:\n"
    semantic_result += semantic_search(query=query, file_pattern=file_pattern, max_results=10,
                                       use_skeleton=False)

    retrieve_result = bm25_result + '\n' + semantic_result
    return retrieve_result


def bm25_retrieve(
        query: str,
        file_pattern: Optional[str] = None,
        similarity_top_k: int = 15
) -> str:
    """Retrieves code snippets from the codebase using the BM25 algorithm based on the provided query, class names, and function names. This function helps in finding relevant code sections that match specific criteria, aiding in code analysis and understanding.

    Args:
        query (Optional[str]): A textual query to search for relevant code snippets. Defaults to an empty string if not provided.
        class_names (list[str]): A list of class names to include in the search query. If None, class names are not included.
        function_names (list[str]): A list of function names to include in the search query. If None, function names are not included.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
        similarity_top_k (int): The number of top similar documents to retrieve based on the BM25 ranking. Defaults to 15.

    Returns:
        str: A formatted string containing the search results, including file paths and the retrieved code snippets (the partial code of a module or the skeleton of the specific module).
    """

    issue_id, instance, structure = get_current_issue_data()
    files, _, _ = get_current_repo_modules()

    repo_playground = get_repo_save_dir()
    repo_dir = setup_full_swebench_repo(instance_data=instance, repo_base_dir=repo_playground)
    persist_dir = os.path.join(
        INDEX_STORE_LOC, get_repo_dir_name(instance["instance_id"])
    )
    workspace = Workspace.from_dirs(repo_dir=repo_dir, index_dir=persist_dir)
    code_index = workspace.code_index
    # bm25_retriever = BM25Retriever.from_persist_dir("plugins/retrieval_tools/bm25_retriever/persist_data/")
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=code_index._docstore,
        similarity_top_k=10,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    # message = f"Search results for query `{query}`:\n"
    message = ""

    all_file_paths = [file[0] for file in files]
    exclude_files = find_matching_files_from_list(all_file_paths, "**/test*/**")
    if file_pattern:
        include_files = find_matching_files_from_list(all_file_paths, file_pattern)
        if not include_files:
            include_files = all_file_paths
            message += f"No files found for file pattern {file_pattern}. Will search all files.\n...\n"
    else:
        include_files = all_file_paths

    retrieved_nodes = bm25_retriever.retrieve(query)
    result_template = """file: {file}
```
{code_content}
```
"""
    # similarity: {score}

    search_result_strs = []
    for node in retrieved_nodes:
        file = node.metadata['file_path']
        if file not in exclude_files and file in include_files:
            if len(node.metadata['span_ids']) == 1 and node.metadata['span_ids'][0] == 'imports':
                continue
            content = get_file_content_(file, return_str=True)
            if not content:
                continue
            result_content = line_wrap_content(content,
                                               [(node.metadata['start_line'], node.metadata['end_line'])])
            search_result_str = result_template.format(
                file=file,
                code_content=result_content
            )

            # code_content = node.node.get_content().strip()
            # if node.metadata['tokens'] <= 100:
            #     search_result_str = result_template.format(
            #         file=file,
            #         # score=node.score,
            #         code_content=code_content
            #     )
            # else:
            #     code_structure = get_skeleton(code_content).strip()
            #     search_result_str = result_template.format(
            #         file=file,
            #         # score=node.score,
            #         code_content=code_structure
            #     )
            search_result_strs.append(search_result_str)
    if search_result_strs:
        search_result_strs = search_result_strs[:5]  # 5 at most
        # message += f'Found {len(search_result_strs)} code spans. The skeleton of each module are as follows.\n\n'
        message += f'Found {len(search_result_strs)} code spans.\n\n'
        return (message + "\n".join(search_result_strs))
    else:
        return 'No locations found.'


def bm25_module_retrieve(
        query: str,
        include_files: Optional[List[str]] = None,
        # file_pattern: Optional[str] = None,
        search_scope: str = 'all',
        similarity_top_k: int = 10,
        # sort_by_type = False
):
    retriever = build_module_retriever(entity_searcher=get_graph_entity_searcher(),
                                       search_scope=search_scope,
                                       similarity_top_k=similarity_top_k)
    try:
        retrieved_nodes = retriever.retrieve(query)
    except IndexError as e:
        logging.warning(f'{e}. Probably because the query `{query}` is too short.')
        return []

    filter_nodes = []
    all_nodes = []
    for node in retrieved_nodes:
        if node.score <= 0:
            continue
        if not include_files or node.text.split(':')[0] in include_files:
            filter_nodes.append(node.text)
        all_nodes.append(node.text)

    if filter_nodes:
        return filter_nodes
    else:
        return all_nodes


def bm25_code_retrieve(
        query_info: QueryInfo,
        # query: str,
        include_files: Optional[List[str]] = None,
        # file_pattern: Optional[str] = None,
        similarity_top_k: int = 10
) -> str:
    """Retrieves code snippets from the codebase using the BM25 algorithm based on the provided query, class names, and function names. This function helps in finding relevant code sections that match specific criteria, aiding in code analysis and understanding.

    Args:
        query (Optional[str]): A textual query to search for relevant code snippets. Defaults to an empty string if not provided.
        class_names (list[str]): A list of class names to include in the search query. If None, class names are not included.
        function_names (list[str]): A list of function names to include in the search query. If None, function names are not included.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.
        similarity_top_k (int): The number of top similar documents to retrieve based on the BM25 ranking. Defaults to 15.

    Returns:
        str: A formatted string containing the search results, including file paths and the retrieved code snippets (the partial code of a module or the skeleton of the specific module).
    """

    issue_id, instance, structure = get_current_issue_data()
    query = query_info.term
    
    # 判断corpus是否存在，不存在则重新生成
    persist_path = os.path.join(BM25_PERSIS_LOC, instance["instance_id"])
    if os.path.exists(f'{persist_path}/corpus.jsonl'):
        # TODO: if similairy_top_k 大于 persit_retriever setting 中的 similairy_top_k
        # 则需要重新生成
        retriever = load_retriever(persist_path)
    else:
        repo_playground = get_repo_save_dir()
        repo_dir = setup_full_swebench_repo(instance_data=instance, repo_base_dir=repo_playground)
        absolute_repo_dir = os.path.abspath(repo_dir)
        retriever = build_code_retriever(absolute_repo_dir, persist_path=persist_path,
                                         similarity_top_k=similarity_top_k)

    # similarity: {score}
    cur_query_results = []
    retrieved_nodes = retriever.retrieve(query)
    for node in retrieved_nodes:
        file = node.metadata['file_path']
        # print(node.metadata)
        if not include_files or file in include_files:
            # 检索到都是 import 信息，则 drop
            # if len(node.metadata['span_ids']) == 1 and node.metadata['span_ids'][0] == 'imports':
            #     continue
            if all([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                # TODO: drop ?
                query_result = QueryResult(query_info=query_info, 
                                           format_mode='code_snippet',
                                           nid=node.metadata['file_path'],
                                           file_path=node.metadata['file_path'],
                                           start_line=node.metadata['start_line'],
                                           end_line=node.metadata['end_line'],
                                           retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                           )
                cur_query_results.append(query_result)
                
            elif any([span_id in ['docstring', 'imports', 'comments'] for span_id in node.metadata['span_ids']]):
                nids = []
                for span_id in node.metadata['span_ids']:
                    nid = f'{file}:{span_id}'
                    searcher = get_graph_entity_searcher()
                    if searcher.has_node(nid):
                        nids.append(nid)
                    # TODO: warning if not find
                    
                node_datas = searcher.get_node_data(nids, return_code_content=True)
                sorted_ndatas = sorted(node_datas, key=lambda x: x['start_line'])
                sorted_nids = [ndata['node_id'] for ndata in sorted_ndatas]
                
                message = ''
                if sorted_nids:
                    if sorted_ndatas[0]['start_line'] < node.metadata['start_line']:
                        nid = sorted_ndatas[0]['node_id']
                        ntype = sorted_ndatas[0]['type']
                        # The code for {ntype} {nid} is incomplete; search {nid} for the full content if needed.
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if sorted_ndatas[-1]['end_line'] > node.metadata['end_line']:
                        nid = sorted_ndatas[-1]['node_id']
                        ntype = sorted_ndatas[-1]['type']
                        message += f"The code for {ntype} `{nid}` is incomplete; search `{nid}` for the full content if needed.\n"
                    if message.strip():
                        message = "Hint: \n"+ message
                
                nids_str = ', '.join([f'`{nid}`' for nid in sorted_nids])
                desc = f"Found {nids_str}."
                query_result = QueryResult(query_info=query_info, 
                                           format_mode='code_snippet',
                                           nid=node.metadata['file_path'],
                                           file_path=node.metadata['file_path'],
                                           start_line=node.metadata['start_line'],
                                           end_line=node.metadata['end_line'],
                                           desc=desc,
                                           message=message,
                                           retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                           )
                
                cur_query_results.append(query_result)
            else:
                for span_id in node.metadata['span_ids']:
                    nid = f'{file}:{span_id}'
                    searcher = get_graph_entity_searcher()
                    if searcher.has_node(nid):
                        ndata = searcher.get_node_data([nid], return_code_content=True)[0]
                        query_result = QueryResult(query_info=query_info, format_mode='preview', 
                                                   nid=ndata['node_id'],
                                                   ntype=ndata['type'],
                                                   start_line=ndata['start_line'],
                                                   end_line=ndata['end_line'],
                                                   retrieve_src=f"Retrieved code content using keyword search (bm25)."
                                                   )
                        cur_query_results.append(query_result)
                    else:
                        continue
        
        cur_query_results = cur_query_results[:5]
        return cur_query_results


def extract_file_to_code(raw_content: str):
    # import re
    # Pattern to extract the file name and code
    pattern = r'([\w\/\.]+)\n```\n(.*?)\n```'

    # Use re.findall to extract all matches (file name and code)
    matches = re.findall(pattern, raw_content, re.DOTALL)

    # Create a dictionary from the extracted file names and code
    file_to_code = {filename: code for filename, code in matches}

    return file_to_code


def semantic_search(
        query: Optional[str] = None,
        code_snippet: Optional[str] = None,
        class_names: list[str] = None,
        function_names: list[str] = None,
        use_skeleton: bool = False,
        file_pattern: Optional[str] = "**/*.py",
        max_results: int = 10,
) -> str:
    """Performs a semantic search over the codebase by incorporating the provided query, code snippet, class names, and function names into a combined search query. This function includes the specified class names and function names in the query before performing the semantic search, enabling a more targeted and relevant search for semantically similar code snippets.

    Args:
        query (Optional[str]): A textual query describing what to search for in the codebase.
        code_snippet (Optional[str]): A code snippet to find semantically similar code in the codebase.
        class_names (list[str]): A list of class names to include in the search query.
        function_names (list[str]): A list of function names to include in the search query.
        file_pattern (Optional[str]): A glob pattern to filter search results to specific file types or directories. If None, the search includes all files.

    Returns:
        str: A formatted string containing the search results, including messages and code snippets of the semantically similar code sections found.
    """

    issue_id, instance, structure = get_current_issue_data()
    repo_playground = get_repo_save_dir()
    repo_dir = setup_full_swebench_repo(instance_data=instance, repo_base_dir=repo_playground)
    persist_dir = os.path.join(INDEX_STORE_LOC, get_repo_dir_name(instance["instance_id"]))
    workspace = Workspace.from_dirs(repo_dir=repo_dir, index_dir=persist_dir)
    # file_context = workspace.create_file_context()
    file_context = workspace.file_context

    # message = f"Searching for query [{query[:20]}...] and file pattern [{file_pattern}]."
    message = ""
    search_response = workspace.code_index.semantic_search(
        query=query,
        code_snippet=code_snippet,
        class_names=class_names,
        function_names=function_names,
        file_pattern=file_pattern,
        max_results=max_results,
    )
    if not search_response.hits:
        message += f"No files found for file pattern {file_pattern}. Will search all files.\n...\n"
        search_response = workspace.code_index.semantic_search(
            query=query,
            code_snippet=code_snippet,
            class_names=class_names,
            function_names=function_names,
            file_pattern=None,
            max_results=max_results,
        )

    for hit in search_response.hits:
        for span in hit.spans:
            file_context.add_span_to_context(
                hit.file_path, span.span_id, tokens=1  # span.tokens
            )
    if file_context.files:
        file_context.expand_context_with_init_spans()
    result_template = """file: {file}
```
...
{code_content}

...
```
"""
    if use_skeleton:
        file_to_search_result = extract_file_to_code(file_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=False,
            exclude_comments=False,
        ))
        search_result_str = f'Found {len(file_to_search_result)} code spans. The skeleton of each module are as follows.\n\n'
        file_to_skeleteons = dict()
        for file, code_content in file_to_search_result.items():
            code_structure = get_skeleton(code_content).strip()
            file_to_skeleteons[file] = code_structure
        search_result_str += '\n'.join([result_template.format(file=file, code_content=code) for file, code in
                                        file_to_skeleteons.items()])
    else:
        search_result_str = search_response.message + '\n'
        search_result_str += file_context.create_prompt(exclude_comments=False)

    return search_result_str


def search_dependency_graph_one_hop(issue_id: str, module_name: str, file_path: str, ntype: str):
    if ntype == 'file':
        nid = file_path
    elif ntype == 'function' or ntype == 'class':
        nid = file_path + ':' + module_name
    else:
        raise NotImplementedError

    G = pickle.load(
        open(f"{DEPENDENCY_GRAPH_LOC}/{issue_id}.pkl", "rb")
    )

    if nid not in G: return None

    searcher = RepoSearcher(G)
    return searcher.one_hop_neighbors(nid, return_data=True)


def search_invoke_and_reference(module_name: str, file_path: str, ntype: str) -> str:
    """Analyzes the dependencies of a specific class, or function, identifying how they interact with other parts of the codebase. This is useful for understanding the broader impact of any changes or issues and ensuring that all related components are accounted for.
    
    Args:
        module_name (str): The name of the module to analyze. This could refer to the name of a class or a function within the codebase. Example values: "class_A", "function_1" or "class_B.function_2".
        file_path (str): The full path to the file where the module resides. This helps in precisely locating the module within the codebase.
        ntype (str): The type of the module being analyzed. Must be one of the following values:
                    - 'class': when analyzing a class's dependencies.
                    - 'function': when analyzing a function's dependencies.
        
    Returns:
        str: A string containing the dependencies for the specified module, which shows other modules that invoke 
             or are invoked by this module. Returns `None` if no dependencies are found.
    Raises:
        NotImplementedError: If the `ntype` is not one of 'class', or 'function'.

    Example:
        search_invoke_and_reference("function_1", "path/to/file1.py", "function")
    """
    graph_context = ""
    graph_item_format = """
### Dependencies for {ntype} `{module}` in `{fname}`:
{dependencies}
"""
    tag_format = """
location: {fname} lines {start_line} - {end_line}
name: {name}
contents: 
{contents}

"""

    issue_id = get_current_issue_id()
    files, classes, functions = get_current_repo_modules()

    # 判断参数的有效性
    if ntype == 'class':
        selected_class = [cls for cls in classes if cls['name'] == module_name and cls['file'] == file_path]
        if not selected_class:
            selected_class = [cls for cls in classes if cls['name'] == module_name]
            if len(selected_class) == 1:
                file_path = selected_class[0]['file']
            else:
                return ""
    elif ntype == 'function' and '.' in module_name:
        class_name = module_name.split('.')[0]
        method_name = module_name.split('.')[-1]

        selected_class = [cls for cls in classes if cls['name'] == class_name and cls['file'] == file_path]
        if selected_class:
            selected_class = selected_class[0]
            selected_method = [method for method in selected_class['methods'] if
                               method['name'] == method_name]
            if not selected_method:
                return ""
        else:
            selected_class = [cls for cls in classes if cls['name'] == module_name]
            if len(selected_class) == 1:
                selected_class = selected_class[0]
                selected_method = [method for method in selected_class['methods'] if
                                   method['name'] == method_name]
                if selected_method:
                    file_path = selected_class['file']
                else:
                    return ""
            else:
                return ""
    else:
        selected_func = [func for func in functions if
                         func['name'] == module_name and func['file'] == file_path]
        if not selected_func:
            selected_func = [func for func in functions if func['name'] == module_name]
            if len(selected_func) == 1:
                file_path = selected_func[0]['file']
            else:
                return ""

    results = search_dependency_graph_one_hop(issue_id, module_name, file_path, ntype)

    if not results:
        return ""
    filter_results = []
    for result in results:
        if result['type'] == 'file' or is_test(result['file_path']):
            continue
        if 'contains' in result['relation']:
            continue
        filter_results.append(result)

    if not filter_results:
        return ""

    code_graph_context = ""
    for result in filter_results:
        f_name = result['file_path']
        m_name = result['module_name']
        content = get_file_content_(f_name, return_str=True)
        if not content:
            continue

        if result['type'] == 'class':
            selected_class = \
                [cls for cls in classes if cls['name'] == result['module_name'] and cls['file'] == f_name][0]
            try:
                init_method = \
                    [method for method in selected_class['methods'] if method['name'] == '__init__'][0]
                m_name = f'{m_name}.__init__'
                start_line = selected_class['start_line']
                end_line = init_method['end_line']
            except:
                start_line = selected_class['start_line']
                end_line = selected_class['end_line']
        else:
            start_line = result['start_line']
            end_line = result['end_line']

        # module_content = content[result['start_line']-1: result['end_line']]
        module_content = line_wrap_content(content, [(start_line, end_line)])
        code_graph_context += tag_format.format(
            fname=f_name,
            name=m_name,
            start_line=start_line,
            end_line=end_line,
            contents=module_content
        )
    graph_context += graph_item_format.format(
        ntype=ntype,
        module=module_name,
        fname=file_path,
        dependencies=code_graph_context
    )
    return graph_context


def search_interactions_among_modules(module_ids: list[str]):
    """Analyze the interactions between specified modules by examining a pre-built dependency graph.
    Args:
        module_ids (list[str]): A list of unique identifiers for modules to analyze. 
                                Each identifier corresponds to either a file or a function/class within a file:
                                - For files, the identifier is the full file path, e.g., 'full_path1/file1.py'.
                                - For functions or classes, the identifier includes the file path and the module name, e.g., 
                                  'full_path1/file1.py (function: MyClass1.entry_function)' or 'full_path1/file1.py (class: MyClass2)'.
    Returns:
        str: A formatted string describing the interactions (edges) between the specified modules in the dependency graph. 
             The format shows relationships between source and target modules, such as:
             'source_file (type: source_module) -> relation -> target_file (type: target_module)'.
             Returns `None` if no interactions are found.
    """

    issue_id, _, structure = get_current_issue_data()
    G = pickle.load(
        open(f"{DEPENDENCY_GRAPH_LOC}/{issue_id}.pkl", "rb")
    )
    searcher = RepoSearcher(G)

    nids = []
    for mid in module_ids:
        if '(' not in mid and mid in G:
            nids.append(mid)
        else:
            module_id, ntype = extract_module_id(mid)
            if module_id and module_id in G:
                nids.append(module_id)

    if not nids:
        return "None"

    # Initialize an empty set to store all the nodes in the paths
    nodes_in_paths = set()

    # Collect all nodes and edges that are part of the paths between pairs of nodes
    for i, node in enumerate(nids):
        for other_node in nids[i + 1:]:
            try:
                # Find the shortest path between the pair
                path = nx.shortest_path(G, source=node, target=other_node)
                nodes_in_paths.update(path)
            except nx.NetworkXNoPath:
                continue

    if nodes_in_paths:
        print(nodes_in_paths)
        edges, node_data = searcher.subgraph(nodes_in_paths)
    else:
        edges, node_data = searcher.subgraph(nids)
    # print(nids, edges)

    edge_str = ""
    edge_template = """{source_nid} -> {relation} -> {tartget_id}\n"""
    for edge in edges:
        source_nid = get_formatted_node_str(edge[0], node_data)
        tartget_id = get_formatted_node_str(edge[1], node_data)

        edge_str += edge_template.format(
            source_nid=source_nid,
            relation=edge[2],
            tartget_id=tartget_id
        )
    return edge_str


def _validate_graph_explorer_inputs(
        start_entities: List[str],
        direction: str = 'downstream',
        traversal_depth: int = 1,
        node_type_filter: Optional[List[str]] = None,
        edge_type_filter: Optional[List[str]] = None,
):
    """evaluate input arguments
    """

    # assert len(invalid_entities) == 0, (
    #     f"Invalid value for `start_entities`: entities {invalid_entities} are not in the repository graph."
    # )
    assert direction in ['downstream', 'upstream', 'both'], (
        "Invalid value for `direction`: Expected one of 'downstream', 'upstream', and 'both'. "
        f"Received: '{direction}'."
    )
    assert traversal_depth == -1 or traversal_depth >= 0, (
        "Invalid value for `traversal_depth`: It must be either -1 or a non-negative integer (>= 0). "
        f"Received: {traversal_depth}."
    )
    if isinstance(node_type_filter, list):
        invalid_ntypes = []
        for ntype in invalid_ntypes:
            if ntype not in VALID_NODE_TYPES:
                invalid_ntypes.append(ntype)
        assert len(
            invalid_ntypes) == 0, f"Invalid node types {invalid_ntypes} in node_type_filter. Expected node type in {VALID_NODE_TYPES}"
    if isinstance(edge_type_filter, list):
        invalid_etypes = []
        for etype in edge_type_filter:
            if etype not in VALID_EDGE_TYPES:
                invalid_etypes.append(etype)
        assert len(
            invalid_etypes) == 0, f"Invalid edge types {invalid_etypes} in edge_type_filter. Expected edge type in {VALID_EDGE_TYPES}"

    graph = get_graph()
    entity_searcher = get_graph_entity_searcher()

    hints = ''
    valid_entities = []
    for i, root in enumerate(start_entities):
        # process node name
        if root != '/':
            root = root.strip('/')
        if root.endswith('.__init__'):
            root = root[:-(len('.__init__'))]

        # validate node name
        if root not in graph:
            # search with bm25
            module_nids = bm25_module_retrieve(query=root)
            module_datas = entity_searcher.get_node_data(module_nids, return_code_content=False)
            if len(module_datas) > 0:
                hints += f'The entity name `{root}` is invalid. Based on your input, here are some candidate entities you might be referring to:\n'
                for module in module_datas[:5]:
                    ntype = module['type']
                    nid = module['node_id']
                    hints += f'{ntype}: `{nid}`\n'
                hints += "Source: Retrieved entity using keyword search (bm25).\n\n"
            else:
                hints += f'The entity name `{root}` is invalid. There are no possible candidate entities in record.\n'
        elif is_test_file(root):
            hints += f'No results for the test entity: `{root}`. Please do not include any test entities.\n\n'
        else:
            valid_entities.append(root)

    return valid_entities, hints


def explore_graph_structure(
        start_entities: List[str],
        direction: str = 'downstream',
        traversal_depth: int = 1,
        entity_type_filter: Optional[List[str]] = None,
        dependency_type_filter: Optional[List[str]] = None,
        # input_node_ids: List[str],
        # direction: str = 'forward',
        # traverse_hop: int = 1,
        # node_type_filter: Optional[List[str]] = None,
        # edge_type_filter: Optional[List[str]] = None,
        # return_code_content: bool = False,
):
    """
    Args:
        start_entities:
        direction:
        traversal_depth:
        entity_type_filter:
        dependency_type_filter:

    Returns:
    """
    start_entities, hints = _validate_graph_explorer_inputs(start_entities, direction, traversal_depth,
                                            entity_type_filter, dependency_type_filter)
    G = get_graph()

    rtn_str = traverse_graph_structure(G, start_entities, direction, traversal_depth,
                                       entity_type_filter, dependency_type_filter)

    if hints.strip():
        rtn_str += "\n\n" + hints
    return rtn_str.strip()


# def explore_repo_structure(start_entity='/', traversal_depth=3):
#     if start_entity != '/':
#         start_entity = start_entity.strip('/')
#
#     instance_id = get_current_issue_id()
#     graph = pickle.load(
#         open(f"{DEPENDENCY_GRAPH_LOC}/{instance_id}.pkl", "rb")
#     )
#     if start_entity not in graph.nodes:
#         return f"Error: The node {start_entity} is not in the graph."
#     if traversal_depth == -1:
#         traversal_depth = 100
#
#     rtn_str = []
#
#     def traverse(node, prefix, is_last, level):
#         if level > traversal_depth:
#             return
#
#         if node == start_entity:
#             rtn_str.append(f"{node}")
#             # print(f"{node}")
#             new_prefix = ''
#         else:
#             connector = '└── ' if is_last else '├── '
#             # print(f"{prefix}{connector}{node}")
#             rtn_str.append(f"{prefix}{connector}{node}")
#             new_prefix = prefix + ('    ' if is_last else '│   ')
#
#         # Stop if the current node is a file (leaf node)
#         # if graph.nodes[node].get('type') == 'file':
#         #     return
#
#         # Traverse neighbors with edge type 'contains' and not test files
#         neigh_ids = []
#         # neighbors = list(graph.neighbors(node))
#         for neighbor in graph.neighbors(node):
#             for key in graph[node][neighbor]:
#                 if graph[node][neighbor][key].get('type') == 'contains':
#                     if not is_test_file(neighbor):
#                         neigh_ids.append(neighbor)
#         for i, neigh_id in enumerate(neigh_ids):
#             is_last_child = (i == len(neigh_ids) - 1)
#             traverse(neigh_id, new_prefix, is_last_child, level + 1)
#
#     traverse(start_entity, '', False, 0)
#     return "\n".join(rtn_str)


def explore_tree_structure(
        start_entities: List[str],
        direction: str = 'downstream',
        traversal_depth: int = 2,
        entity_type_filter: Optional[List[str]] = None,
        dependency_type_filter: Optional[List[str]] = None,
):
    """Analyzes and displays the dependency structure around specified entities in a code graph.

    This function searches and presents relationships and dependencies for the specified entities (such as classes, functions, files, or directories) in a code graph.
    It explores how the input entities relate to others, using defined types of dependencies, including 'contains', 'imports', 'invokes' and 'inherits'.
    The search can be controlled to traverse upstream (exploring dependencies that entities rely on) or downstream (exploring how entities impact others), with optional limits on traversal depth and filters for entity and dependency types.

    Example Usage:
    1. Exploring Outward Dependencies:
        ```
        get_local_structure(
            start_entities=['src/module_a.py:ClassA'],
            direction='downstream',
            traversal_depth=2,
            entity_type_filter=['class', 'function'],
            dependency_type_filter=['invokes', 'imports']
        )
        ```
        This retrieves the dependencies of `ClassA` up to 2 levels deep, focusing only on classes and functions with 'invokes' and 'imports' relationships.

    2. Exploring Inward Dependencies:
        ```
        get_local_structure(
            start_entities=['src/module_b.py:FunctionY'],
            direction='upstream',
            traversal_depth=-1
        )
        ```
        This finds all entities that depend on `FunctionY` without restricting the traversal depth.

    Notes:
    * Traversal Control: The `traversal_depth` parameter specifies how deep the function should explore the graph starting from the input entities.
    * Filtering: Use `entity_type_filter` and `dependency_type_filter` to narrow down the scope of the search, focusing on specific entity types and relationships.
    * Graph Context: The function operates on a pre-built code graph containing entities (e.g., files, classes and functions) and dependencies representing their interactions and relationships.

    Parameters:
    ----------
    start_entities : list[str]
        List of entities (e.g., class, function, file, or directory paths) to begin the search from.
        - Entities representing classes or functions must be formatted as "file_path:QualifiedName"
          (e.g., `interface/C.py:C.method_a.inner_func`).
        - For files or directories, provide only the file or directory path (e.g., `src/module_a.py` or `src/`).

    direction : str, optional
        Direction of traversal in the code graph; allowed options are:
        - 'upstream': Traversal to explore dependencies that the specified entities rely on (how they depend on others).
        - 'downstream': Traversal to explore the effects or interactions of the specified entities on others
          (how others depend on them).
        - 'both': Traversal in both directions.
        Default is 'downstream'.

    traversal_depth : int, optional
        Maximum depth of traversal. A value of -1 indicates unlimited depth (subject to a maximum limit).
        Must be either `-1` or a non-negative integer (≥ 0).
        Default is 2.

    entity_type_filter : list[str], optional
        List of entity types (e.g., 'class', 'function', 'file', 'directory') to include in the traversal.
        If None, all entity types are included.
        Default is None.

    dependency_type_filter : list[str], optional
        List of dependency types (e.g., 'contains', 'imports', 'invokes', 'inherits') to include in the traversal.
        If None, all dependency types are included.
        Default is None.

    Returns:
    -------
    result : object
        An object representing the traversal results, which includes discovered entities and their dependencies.
    """
    start_entities, hints = _validate_graph_explorer_inputs(start_entities, direction, traversal_depth,
                                                            entity_type_filter, dependency_type_filter)
    G = get_graph()

    # return_json = True
    return_json = False
    if return_json:
        rtns = {node: traverse_json_structure(G, node, direction, traversal_depth, entity_type_filter,
                                              dependency_type_filter)
                for node in start_entities}
        rtn_str = json.dumps(rtns)
    else:
        rtns = [traverse_tree_structure(G, node, direction, traversal_depth, entity_type_filter,
                                        dependency_type_filter)
                for node in start_entities]
        rtn_str = "\n\n".join(rtns)
        
    if hints.strip():
        rtn_str += "\n\n" + hints
    return rtn_str.strip()


__all__ = [
    'get_repo_structure',
    'get_directory_structure',
    'get_file_content',
    'get_file_structures',
    'search_class',
    'search_class_structures',
    'search_method',

    # 'search_in_repo',
    'search_code_snippets',
    'explore_graph_structure',
    'explore_tree_structure',

    # 'explore_repo_structure',
    # 'search_invoke_and_reference',
    # 'search_interactions_among_modules'
]
