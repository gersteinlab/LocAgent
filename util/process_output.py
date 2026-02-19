import re
import json
import logging
import collections
from collections import Counter
from util.benchmark.parse_python_file import (
    parse_global_var_from_code, is_global_var
)
from dependency_graph import RepoEntitySearcher
from dependency_graph.build_graph import (
    NODE_TYPE_FILE, NODE_TYPE_FUNCTION, NODE_TYPE_CLASS
)
import pickle
import os
GRAPH_INDEX_DIR = os.environ.get("GRAPH_INDEX_DIR", "index_data/graph_index")


def parse_raw_loc_output(raw_output, valid_files):
    valid_top_folder = []
    for fn in valid_files:
        folder = fn.split('/')[0]
        if folder not in valid_top_folder:
            valid_top_folder.append(folder)
    
    # Remove the triple backticks and any surrounding whitespace
    raw_output = raw_output.strip('` \n')
    file_list, loc_edit_list = [], []
    
    current_file = None
    # Split the input data into lines
    lines = raw_output.strip().split('\n')
    for line in lines:
        line = line.strip().strip(':').strip()
        if not line:
            continue  # Skip empty lines

        if any(line.endswith(ext) for ext in _valid_extensions(valid_files)) or '/' in line:
            fn = extract_file_path(line, valid_top_folder, valid_files)
            if not fn or fn not in valid_files:
                current_file = None
                continue

            current_file = fn
            if current_file not in file_list:
                file_list.append(current_file)

        elif line and any(
            line.startswith(w)
            for w in ["function:", "class:", 'method:', 
                      "variable:", 'variables:', "line:", "lines:"]
        ):
            loc = f'{current_file}:{line.strip()}'
            if loc not in loc_edit_list:
                loc_edit_list.append(loc)
            # if current_file and line not in loc_edit_dict[current_file]:
            #     loc_edit_dict[current_file].append(line)

    return file_list, loc_edit_list


def get_loc_results_from_raw_outputs(instance_id, raw_outputs, include_variable=False, graph=None):
    if graph is None:
        if instance_id is None:
            raise ValueError("instance_id is required when graph is not provided")
        G = pickle.load(open(f"{GRAPH_INDEX_DIR}/{instance_id}.pkl", "rb"))
    else:
        G = graph
    searcher = RepoEntitySearcher(G)
    all_files = searcher.get_all_nodes_by_type(NODE_TYPE_FILE)
    valid_files = [file['name'] for file in all_files]
    
    all_found_files = [[] for _ in range(len(raw_outputs))]
    all_found_modules = [[] for _ in range(len(raw_outputs))]
    all_found_entities = [[] for _ in range(len(raw_outputs))]
    for i, sample in enumerate(raw_outputs):
        found_files, found_edit_locs = parse_raw_loc_output(sample, valid_files)
        all_found_files[i] = found_files
        edit_entities = get_edit_entities_from_raw_locs(found_edit_locs, searcher,
                                                        include_variable=include_variable)
        
        filtered_edit_entities = []
        edit_modules = []
        for entity in edit_entities:
            # if entity.endswith('.__init__'):
            #     entity = entity[:(len(entity)-len('.__init__'))]
            if searcher.has_node(entity):
                entity_data = searcher.get_node_data([entity])[0]
                if entity_data['type'] == NODE_TYPE_FUNCTION:
                    filtered_edit_entities.append(entity)
            else:
                continue
            
            if '.' in entity.split(':')[-1]:
                entity = '.'.join(entity.split('.')[:-1])
                if searcher.has_node(entity):
                    if entity not in edit_modules:
                        edit_modules.append(entity)
            else:
                if entity not in edit_modules:
                    edit_modules.append(entity)
            
        all_found_entities[i] = filtered_edit_entities
        all_found_modules[i] = edit_modules
    return all_found_files, all_found_modules, all_found_entities


def _valid_extensions(valid_files):
    exts = set()
    for fn in valid_files:
        if '.' in fn:
            exts.add('.' + fn.split('.')[-1])
    return exts or {'.py'}


def extract_file_path(line, valid_folders, valid_files):
    """
    Extracts a file path from a given line of text.

    Parameters:
    - line (str): A line of text that may contain a file path.
    - valid_folders (list[str]): Top-level folders that are valid for the repo.
    - valid_files (list[str]): Full valid file paths.

    Returns:
    - str or None: The extracted file path if found; otherwise, None.
    """
    exts = _valid_extensions(valid_files)
    ext_pattern = '|'.join(re.escape(ext) for ext in sorted(exts))
    pattern = rf'[\\w\\./-]+(?:{ext_pattern})'

    match = re.search(pattern, line)
    if not match:
        return None

    matched_fp = match.group(0)
    start_index = len(matched_fp)
    for folder in valid_folders:
        if f'{folder}/' in matched_fp:
            cur_start_index = matched_fp.index(f'{folder}/')
            if cur_start_index < start_index:
                start_index = cur_start_index
    candidate = matched_fp[start_index:] if start_index < len(matched_fp) else matched_fp

    if candidate in valid_files:
        return candidate
    return None


def merge_sample_locations(found_files, found_modules, found_entities, ranking_method='majority'):
    
    def rank_locs(found_locs, ranking_method="majority"):
        flat_locs = [loc for sublist in found_locs for loc in sublist]
        # unique_files = list(set(flat_files))
        locs_weights = collections.defaultdict(float)
        # ranked_locs = list()
        
        if ranking_method == "majority":
            """Rank files based on their frequency of occurrence"""
            loc_counts = Counter(flat_locs)
            for loc, count in loc_counts.items():
                locs_weights[loc] = count
        
        elif ranking_method == "mrr":
            """Rank files based on Mean Reciprocal Rank (MRR) of their edit locations"""
            # Calculate MRR for the edit locations: sum of (1 / rank)
            for sample_locs in found_locs:
                for rank, loc in enumerate(sample_locs, start=1):
                    locs_weights[loc] += 1 / rank
        
        # Rank the files based on the selected ranking method
        ranked_loc_weights = sorted(locs_weights.items(), key=lambda x: x[1], reverse=True)
        ranked_locs = [file for file, _ in ranked_loc_weights]
        return ranked_locs, ranked_loc_weights

    # Rank files
    ranked_files, file_weights = rank_locs(found_files, ranking_method)
    ranked_modules, module_weights = rank_locs(found_modules, ranking_method)
    ranked_funcs, func_weights = rank_locs(found_entities, ranking_method)
    
    return ranked_files, ranked_modules, ranked_funcs
    

# def get_edit_modules_from_file_to_dict(pred_files, file_to_edit_locs, structure, keep_whole_class=False):
def get_edit_entities_from_raw_locs(found_edit_locs, 
                                    searcher: RepoEntitySearcher,
                                    ranking_method='majority',
                                    include_variable=False,
                                    ):
    # topn locs
    found_edit_entities = []
    current_class_name = ""
    prev_file_name = ""
    for i, edit_loc in enumerate(found_edit_locs):
        pred_file = edit_loc.split(':')[0].strip()
        if prev_file_name and prev_file_name != pred_file:
            current_class_name = ""
        prev_file_name = pred_file
        
        loc = ':'.join(edit_loc.split(':')[1:]).strip()
        # i = pred_files.index(pred_file)
        
        # get file content -> parse global var
        if searcher.has_node(pred_file):
            pred_file_content = searcher.G.nodes[pred_file]['code']
            global_vars = parse_global_var_from_code(pred_file_content)
        else:
            continue
        
        if loc.startswith("line:") or loc.startswith("lines:"):
            loc = loc.split(":")[1].strip()
            pred_lines = []
            # Regular expression to match different line formats
            # match = re.match(r"\s*(\d+)\s*[-ｰ]?\s*(\d+)?", loc)
            matches = re.findall(r'\s*(\d+)(?:-(\d+))?', loc)
            for match in matches:
                start_line = max(1, int(match[0]))
                end_line = int(match[1]) if match[1] else start_line
                end_line = min(len(pred_file_content.splitlines()), end_line)
                pred_lines += list(range(start_line, end_line+1))
            if not matches:
                loc = loc.split()[0]
                try:
                    pred_lines.append(int(loc.strip()))
                except:
                    logging.debug(f'line {loc} not found')
            
            pred_lines = list(set(pred_lines))
            pred_lines.sort()
            cur_found_modules = get_modules_from_line_numbers(pred_lines, pred_file, searcher, 
                                                              global_vars,
                                                              include_variable=include_variable)
            for cmodule in cur_found_modules:
                if cmodule['type'] == NODE_TYPE_CLASS:
                    current_class_name = cmodule.split(':')[-1].strip()
                    
                if cmodule['type'] == NODE_TYPE_FUNCTION:
                    found_edit_entities.append(cmodule['name'])
        
        # handle cases like "class: MyClass"
        elif loc.startswith("class:") and "." not in loc:
            loc = loc[len("class:") :].strip()
            loc = loc.split()[0]
            module_id = f'{pred_file}:{loc.strip()}'
            if module_id in searcher.G:
                current_class_name = loc
            else:
                logging.info(f"{loc} class could not be found")
                
        elif loc.startswith("function: ") or loc.startswith("method: ") or "." in loc:
            full_loc = loc
            loc = loc.split(":", 1)[-1].strip('() ')
            loc = loc.split()[0]

            # handle cases like "function: MyClass.my_method"/ "class: MyClass.my_method"
            # for cases like "function: MyClass.my_method.inner_method", ignore "inner_method"
            if "." in loc:
                # assume its a method within a class
                class_name = loc.split(".")[0]
                method_name = loc.split(".")[1]
                # if method_name == '__init__':
                #     _module_id = f'{pred_file}:{class_name}'
                
                module_id = f'{pred_file}:{class_name}.{method_name}'
                if module_id in searcher.G:
                    found_edit_entities.append(module_id)
                    continue
                else:
                    logging.debug(f"{full_loc} method could not be found")
                    
            # directly search for the function 'loc'
            if f"{pred_file}:{loc}" in searcher.G:
                found_edit_entities.append(f"{pred_file}:{loc}")
            # relevant_function = get_function_by_name(loc, pred_file, functions=functions)
            else:
                logging.debug(f"{loc} function could not be found")
                
                if current_class_name != "":
                    # check if its a method
                    if f"{pred_file}:{current_class_name}" in searcher.G:
                        potential_class = searcher.get_node_data([f"{pred_file}:{current_class_name}"])[0]
                        if potential_class['type'] == NODE_TYPE_CLASS:
                            _module_id = f"{pred_file}:{current_class_name}.{loc}"
                            if _module_id in searcher.G:
                                found_edit_entities.append(_module_id)
                else:
                    if loc in searcher.global_name_dict:
                        nids = searcher.global_name_dict[loc]
                        cadidate_nids = []
                        for nid in nids:
                            if nid.startswith(pred_file):
                                cadidate_nids.append(nid)
                        if len(cadidate_nids) == 1:
                            found_edit_entities.append(cadidate_nids[0])
            # else:
            #     found_edit_modules.append(f'{pred_file}:{loc}')
        # - end identify function -
        
        elif include_variable and loc.startswith(("variable:", "variables:")):
            vars = loc.split(':')[-1].strip().replace(',', ' ').split()
            # print(vars)
            for v in vars:
                if global_vars and v in global_vars:
                    # if f'{pred_file}:{v}' not in found_edit_modules:
                    found_edit_entities.append(f'{pred_file}:{v}')
        else:
            if loc.strip():
                logging.info(f"loc {loc} not recognised")
    
    
    loc_weights = collections.defaultdict(float)
    # Apply the selected merging method
    if ranking_method == "majority":
        # Majority Voting: Count the frequency of each edit location
        loc_counts = Counter(found_edit_entities)
        for loc, count in loc_counts.items():
            loc_weights[loc] = count
    elif ranking_method == "mrr":
        for rank, loc in enumerate(found_edit_entities, start=1):
            # Calculate MRR for edit locations
            loc_weights[loc] += 1 / rank
            
    # Sort edit locations based on weight
    ranked_loc_weights = sorted(loc_weights.items(), key=lambda x: x[1], reverse=True)
    res_edit_entities = [loc for loc, _ in ranked_loc_weights]
    # found_edit_module_loc = [['\n'.join(modules)] for modules in found_edit_modules]
    # import pdb; pdb.set_trace()
    return res_edit_entities


def get_modules_from_line_numbers(line_numbers, 
                                  pred_file, 
                                  searcher: RepoEntitySearcher,
                                  global_vars: dict=None,
                                  include_variable: bool = False,
                                  ):
    found_mnames, found_mnodes = [], []
    cur_module_end_line = None
    for line in line_numbers:
        # TODO: check if global var
        # if include_variable and global_vars:
        #     variable = is_global_var(line, global_vars)
        #     if variable and variable not in found_modules:
        #         # found_modules.append(f"variable: {variable}")
        #         found_modules.append(f"{pred_file}:{variable}")
        #         continue
        if cur_module_end_line and line <= cur_module_end_line:
            continue
        module, cur_module_end_line = get_module_from_line_number(line, pred_file, searcher)
        if module and module['name'] not in found_mnames:
            found_mnames.append(module['name'])
            found_mnodes.append(module)
    return found_mnodes


def get_module_from_line_number(line, file_path, searcher):
    assert file_path in searcher.G.nodes
    file_node = searcher.get_node_data([file_path])[0]
    cur_start_line = file_node['start_line']
    cur_end_line = file_node['end_line']
    cur_node = None
    
    for nid in searcher.G.nodes():
        # if not nid.startswith(file_path) or ':' not in nid:
        #     continue
        node = searcher.G.nodes[nid]
        if node['type'] != NODE_TYPE_FUNCTION: continue
        if 'start_line' in node and 'end_line' in node:
            if node['start_line'] < cur_start_line or node['end_line'] > cur_end_line:
                continue
            if line >= node['start_line'] and line <= node['end_line']:
                cur_node = node
                cur_node['name'] = nid
                cur_start_line = node['start_line']
                cur_end_line = node['end_line']
    if cur_node:
        return (cur_node, cur_end_line)
    return (None, None)
