import re
import json
import logging
from copy import deepcopy
import collections
from collections import Counter
from utils.benchmark.parse_oracle_patch import is_global_var
from agentless.util.parse_global_var import parse_global_var_from_code
from agentless.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
)
from typing import List, Optional, Union, Tuple, Dict
from graph_encoder.dependency_graph import RepoEntitySearcher
import pickle
import os
DEPENDENCY_GRAPH_LOC = os.environ.get("DEPENDENCY_GRAPH_LOC")

def add_new_file(new_file, valid_files, found_files=None):
    if new_file in valid_files and new_file not in found_files:
        found_files.append(new_file)
    return found_files


def get_loc_related_dict_from_raw_output(raw_output, valid_files, found_files=None, loc_dict=None):
    assert valid_files
    # Remove the triple backticks and any surrounding whitespace
    raw_output = raw_output.strip('` \n')

    # Initialize lists
    if not found_files:
        found_files = []
    if not loc_dict:
        loc_dict = {}

    # Split the input data into lines
    lines = raw_output.strip().split('\n')

    for line in lines:
        line = line.strip().strip(':').strip()
        if not line:
            current_file = None
            continue  # Skip empty lines

        if line.endswith('.py'):
            if line not in valid_files:
                current_file = None
                continue
            # It's a file name
            current_file = line
            if current_file not in found_files:
                found_files.append(current_file)
            if current_file not in loc_dict:
                loc_dict[current_file] = []
        elif line.startswith(('function:', 'class:')):
            # It's a function or class definition
            if current_file:
                loc_dict[current_file].append(line)
    return found_files, loc_dict


def get_additional_artifact_loc_related_from_dict(found_files, found_related_locs):
    files = [f for f in found_files if f in found_related_locs]
    output = "```\n"

    for file, locs in zip(files, found_related_locs):
        output += f"{file}\n"
        for loc in locs:
            output += f"{loc}\n"
        output += "\n"
    output += "```"

    additional_artifact_loc_related = [{"raw_output": output}]
    return additional_artifact_loc_related


def parse_raw_loc_output(raw_output, valid_files, file_list=None, loc_edit_dict=None):
    valid_top_folder = []
    for fn in valid_files:
        folder = fn.split('/')[0]
        if folder not in valid_top_folder:
            valid_top_folder.append(folder)
    
    # Remove the triple backticks and any surrounding whitespace
    raw_output = raw_output.strip('` \n')
    
    # Initialize lists
    if not file_list:
        file_list = []
    if not loc_edit_dict:
        # Initialize the dictionary to store the edit file information
        loc_edit_dict = collections.defaultdict(list)
    
    current_file = None
    # Split the input data into lines
    lines = raw_output.strip().split('\n')
    for line in lines:
        line = line.strip().strip(':').strip()
        if not line:
            continue  # Skip empty lines

        if line.endswith('.py'):
            fn = extract_python_file_path(line, valid_top_folder)
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
            if current_file and line not in loc_edit_dict[current_file]:
                loc_edit_dict[current_file].append(line)

    return file_list, loc_edit_dict


def parse_raw_loc_output_v2(raw_output, valid_files):
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

        if line.endswith('.py'):
            fn = extract_python_file_path(line, valid_top_folder)
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


def convert_to_loc_edit_list(loc_edit_dict, file_list):
    
    loc_edit_list = [[] for _ in range(len(file_list))]
    for i, file in enumerate(file_list):
        if file in loc_edit_dict:
            loc_edit_list[i] = ['\n'.join(loc_edit_dict[file])]
        else:
            loc_edit_list[i] = [""]
                
    return loc_edit_list


def get_loc_results_from_raw_outputs(raw_outputs, valid_files):
    all_found_files = []
    all_found_edit_locs = [[] for _ in range(len(raw_outputs))]
        
    for i, sample in enumerate(raw_outputs):
        file_list, loc_edit_dict = parse_raw_loc_output(sample, valid_files)
        all_found_edit_locs[i] = convert_to_loc_edit_list(loc_edit_dict, file_list)
        all_found_files.append(file_list)
        
    return all_found_files, all_found_edit_locs


def get_loc_results_from_raw_outputs_v2(instance_id, raw_outputs, valid_files, structure, ignore_variable=False):
    all_found_files = [[] for _ in range(len(raw_outputs))]
    all_found_modules = [[] for _ in range(len(raw_outputs))]
    all_found_entities = [[] for _ in range(len(raw_outputs))]
    for i, sample in enumerate(raw_outputs):
        found_files, found_edit_locs = parse_raw_loc_output_v2(sample, valid_files)
        all_found_files[i] = found_files
        # all_found_locs.append(found_edit_locs)
        edit_entities = get_edit_modules_from_file_to_dict(found_edit_locs, structure, keep_whole_class=False, ignore_variable=ignore_variable)
        filtered_edit_entities = []
        edit_modules = []
        G = pickle.load(
                open(f"{DEPENDENCY_GRAPH_LOC}/{instance_id}.pkl", "rb")
            )
        searcher = RepoEntitySearcher(G)
        for entity in edit_entities:            
            if entity.endswith('.__init__'):
                entity = entity[:(len(entity)-len('.__init__'))]
                if searcher.has_node(entity):
                    filtered_edit_entities.append(entity)
                    
            if not searcher.has_node(entity):
                continue
            
            # entity_data = searcher.get_node_data([entity])[0]
            # if entity_data['type'] == 'function':
            filtered_edit_entities.append(entity)
            
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


def get_loc_edit_dict_from_raw_sample_output(data, valid_files, file_list=None, loc_related_dict=None, loc_edit_dict=None):
    valid_top_folder = []
    for fn in valid_files:
        folder = fn.split('/')[0]
        if folder not in valid_top_folder:
            valid_top_folder.append(folder)
    
    # Remove the triple backticks and any surrounding whitespace
    data = data.strip('` \n')

    # Initialize lists
    if not file_list:
        file_list = []
    if not loc_related_dict:
        loc_related_dict = dict()
    if not loc_edit_dict:
        # Initialize the dictionary to store the edit file information
        loc_edit_dict = dict()

    current_file = None
    current_related = None
    # current_data = None
    # Split the input data into lines
    lines = data.strip().split('\n')
    for line in lines:
        line = line.strip().strip(':').strip()
        if not line:
            # current_file = None
            # current_related = None
            # current_data = []
            continue  # Skip empty lines

        if line.endswith('.py'):
            fn = extract_python_file_path(line, valid_top_folder)
            if not fn or fn not in valid_files:
                current_file = None
                current_related = None
                continue

            current_file = fn
            current_related = None
            if current_file not in file_list:
                file_list.append(current_file)
            if current_file not in loc_related_dict:
                loc_related_dict[current_file] = []
            if current_file not in loc_edit_dict:
                loc_edit_dict[current_file] = {}

        elif line.startswith(('function:', 'class:', 'method:', 'variable:', 'variables:')):
            if current_file:
                current_related = line
                if current_related not in loc_related_dict[current_file]:
                    loc_related_dict[current_file].append(current_related)
                if current_related not in loc_edit_dict[current_file]:
                    loc_edit_dict[current_file][current_related] = []
        elif line.startswith('line'):
            # It's part of the function/class/line data
            if current_file and current_related:
                loc_edit_dict[current_file][current_related].append(line)
            elif current_file:
                if '' not in loc_edit_dict[current_file]:
                    loc_edit_dict[current_file][''] = []
                loc_edit_dict[current_file][''].append(line)
    return file_list, loc_related_dict, loc_edit_dict


# def get_loc_edit_dict_from_raw_output(raw_output, valid_files, file_list=None, loc_related_dict=None, all_results=None):
#     # all_results = [dict() for i in range(raw_output)]
#     found_files = []
#     if not all_results:
#         all_results = [dict() for i in range(len(raw_output))]
#     else:
#         assert len(all_results) == len(raw_output)
#     all_loc_related_dict = []
        
#     for i, sample in enumerate(raw_output):
#         file_list, loc_related_dict, loc_edit_dict = get_loc_edit_dict_from_raw_sample_output(
#             sample, valid_files, 
#             # file_list, loc_related_dict, 
#             loc_edit_dict=all_results[i]
#         )  
#         all_results[i] = loc_edit_dict
#         found_files.append(file_list)
#         all_loc_related_dict.append(loc_related_dict) # TODO: process the loc_related variables
#     return found_files, all_loc_related_dict, all_results


def convert_to_loc_related_list(loc_related_dict, file_list):
    loc_related_list = []
    for file in file_list:
        if file in loc_related_dict:
            loc_related_list.append(['\n'.join(loc_related_dict[file])])
        else:
            loc_related_list.append([""])
    return loc_related_list


# def convert_to_loc_edit_list(loc_edit_dict, file_list):
#     loc_edit_list = []
#     for i, sample in enumerate(loc_edit_dict):
#         sample_list = []
#         for file in file_list[i]:
#             data = []
#             if file in sample:
#                 for modual in sample[file]:
#                     data.append(modual)
#                     data += sample[file][modual]
#                 sample_list.append(['\n'.join(data)])
#             else:
#                 sample_list.append([""])
#         loc_edit_list.append(sample_list)
#     return loc_edit_list


def extract_python_file_path(line, valid_folders):
    """
    Extracts the Python file path from a given line of text.

    Parameters:
    - line (str): A line of text that may contain a Python file path.

    Returns:
    - str or None: The extracted Python file path if found; otherwise, None.
    """
    # Define a regular expression pattern to match file paths ending with .py
    # The pattern looks for sequences of characters that can include letters, numbers,
    # underscores, hyphens, dots, or slashes, ending with '.py'
    pattern = r'[\w\./-]+\.py'

    # Search for the pattern in the line
    match = re.search(pattern, line)

    if match:
        matched_fp = match.group(0)
        start_index = len(matched_fp)
        for folder in valid_folders:
            if f'{folder}/' in matched_fp:
                cur_start_index = matched_fp.index(f'{folder}/')
                if cur_start_index < start_index:
                    start_index = cur_start_index
        if start_index < len(matched_fp):
            return matched_fp[start_index:] # Return the max matched file path
        return None
    else:
        return None  # Return None if no match is found
    

def extract_result(summary):
    pattern = r"```(.*?)```"
    match = re.search(pattern, summary, re.DOTALL)

    # Extract and format the result if a match is found
    if match:
        result = f"```{match.group(1)}```"
    else:
        result = ""
        print("No match found")
    return result


def merge_sample_locations_v2(found_files, found_modules, found_entities, ranking_method='majority'):
    
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


def merge_sample_locations(found_files, found_edit_locs, ranking_method='majority'):
    found_edit_locs_dict = [dict() for _ in found_files]
    for i, sample_files in enumerate(found_files):
        for j, f in enumerate(sample_files):
            if len(found_edit_locs) > i and len(found_edit_locs[i]) > j:
                found_edit_locs_dict[i][f] = found_edit_locs[i][j]
    
    def rank_edit_locs(ranked_files, found_edit_locs, ranking_method="mrr"):
        """Merge edit locations using Majority Voting or Mean Reciprocal Rank (MRR)"""
        
        merged_edit_locs = {}
        
        for r_file in ranked_files:
            loc_weights = collections.defaultdict(float)
            
            all_edit_locs = []
            # Apply the selected merging method
            if ranking_method == "majority":
                # Majority Voting: Count the frequency of each edit location
                for sample in found_edit_locs:
                    if r_file in sample:
                        for loc in sample[r_file]:
                            # Split the edit locations by '\n' and extend the list
                            all_edit_locs.extend(loc.split('\n'))
                    
                loc_counts = Counter(all_edit_locs)
                for loc, count in loc_counts.items():
                    loc_weights[loc] = count
                
            elif ranking_method == "mrr":
                sample_locs = []
                for sample in found_edit_locs:
                        if r_file in sample:
                            for loc in sample[r_file]:
                                sample_locs.extend(loc.split('\n'))
                            
                            # Calculate MRR for edit locations
                            for rank, loc in enumerate(sample_locs, start=1):
                                loc_weights[loc] += 1 / rank
                    
            # Sort edit locations based on weight
            ranked_loc_weights = sorted(loc_weights.items(), key=lambda x: x[1], reverse=True)
            
            # Store the merged and ranked edit locations in the dictionary
            merged_edit_locs[r_file] = [loc for loc, _ in ranked_loc_weights]
            
        return merged_edit_locs, ranked_loc_weights

    def rank_files(found_files, ranking_method="mrr"):
        flat_files = [file for sublist in found_files for file in sublist]
        # unique_files = list(set(flat_files))
        file_weights = collections.defaultdict(float)
        ranked_files = list()
        
        if ranking_method == "majority":
            """Rank files based on their frequency of occurrence"""
            file_counts = Counter(flat_files)
            for file, count in file_counts.items():
                file_weights[file] = count
        
        elif ranking_method == "mrr":
            """Rank files based on Mean Reciprocal Rank (MRR) of their edit locations"""
            # Calculate MRR for the edit locations: sum of (1 / rank)
            for sample_files in found_files:
                for rank, file in enumerate(sample_files, start=1):
                    file_weights[file] += 1 / rank
        
        # Rank the files based on the selected ranking method
        ranked_file_weights = sorted(file_weights.items(), key=lambda x: x[1], reverse=True)
        ranked_files = [file for file, _ in ranked_file_weights]
        return ranked_files, ranked_file_weights

    # Rank files
    ranked_files, file_weights = rank_files(found_files, ranking_method)
    ranked_edit_locs, ranked_loc_weights = rank_edit_locs(ranked_files, found_edit_locs_dict, ranking_method)
    
    ranked_loc_edit_list = convert_to_loc_edit_list(ranked_edit_locs, ranked_files)
    return ranked_files, ranked_loc_edit_list


def parse_keyword_json_obj(raw_output_str):
    # Regular expression pattern to match the JSON arrays
    pattern = r'\[\s*\{\s*"keyword":.*?\}\s*\]'

    # Find all matches in the text
    matches = re.findall(pattern, raw_output_str, re.DOTALL)

    # List to hold all extracted JSON objects
    all_objects = []

    # Iterate over the matches and parse the JSON
    for match in matches:
        try:
            json_obj = json.loads(match) # list
            for j in json_obj:
                reduplicated = False
                for obj in all_objects:
                    # print(type(obj), type(json_obj))
                    if obj['keyword'] == j['keyword'] and obj['possible_file_path'] == j['possible_file_path']:
                        obj['possible_line_numbers'] = list(set(obj['possible_line_numbers']) | set(j['possible_line_numbers']))
                        reduplicated = True
                        break
                if not reduplicated:
                    all_objects.append(j)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue

    return all_objects


def get_class_by_name(name: str, file_name: str, classes: List[Dict]):
    relevant_class = [
                        clazz
                        for clazz in classes
                        if clazz["file"] == file_name and clazz["name"] == name
                    ]
    return relevant_class


def get_function_by_name(func_name, file_name, functions=None, cur_class=None, classes=None):
    assert functions or cur_class or classes
    if functions:
        relevant_func = [ func
                            for func in functions
                            if func["file"] == file_name and func["name"] == func_name
                         ]
        return relevant_func
    
    if cur_class:
        # check if its a method
        relevant_method = [
            method
            for method in cur_class[0]["methods"]
            if method["name"] == func_name
        ]
        # if len(relevant_method) == 0:
        #     logging.info(f"{func_name} method could not be found")
        # else:
        return relevant_method
        
    if classes:
        # look for it in any class
        relevant_method = []
        cls_n = None
        for clazz in classes:
            if clazz["file"] == file_name:
                relevant_method.extend(
                    [
                        method
                        for method in clazz["methods"]
                        if method["name"] == func_name
                    ]
                )
                if relevant_method and not cls_n:
                    cls_n = clazz['name']
        return (cls_n, relevant_method)
    
        # if len(relevant_method) == 1:
        #     _module_name = f'function: {cls_n}.{func_name}'
        #     found_edit_modules[i].append(_module_name)
    

# def get_edit_modules_from_file_to_dict(pred_files, file_to_edit_locs, structure, keep_whole_class=False):
def get_edit_modules_from_file_to_dict(found_edit_locs, structure, 
                                       keep_whole_class=False, 
                                       ranking_method='majority',
                                       ignore_variable=False,
                                       ):
    # topn locs
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    if not files or not classes or not functions:
        return []
    # found_edit_modules = [[] for _ in range(len(found_edit_locs))]
    found_edit_modules = []
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
        pred_file_content = ""
        for file_content in files:
            if file_content[0] == pred_file:
                content = "\n".join(file_content[1])
                pred_file_content = content
                break
        if pred_file_content:
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
            cur_found_modules = get_modules_from_line_numbers(pred_lines, pred_file, structure, 
                                                                global_vars, keep_whole_class, 
                                                                ignore_variable=ignore_variable)
            for cmodule in cur_found_modules:
                if cmodule.startswith('class'):
                    current_class_name = cmodule.split(':')[-1].strip()
                module_id = f'{pred_file}:{cmodule.split(':')[-1].strip()}'
                # if module_id not in found_edit_modules:
                found_edit_modules.append(module_id)
        
        # handle cases like "class: MyClass"
        elif loc.startswith("class:") and "." not in loc:
            loc = loc[len("class:") :].strip()
            loc = loc.split()[0]
            
            relevant_class = get_class_by_name(loc, pred_file, classes)
            if len(relevant_class) == 0:
                logging.info(f"{loc} class could not be found")
            else:
                module_id = f'{pred_file}:{loc.strip()}'
                # if module_id not in found_edit_modules:
                found_edit_modules.append(module_id)
                    # found_edit_modules[i].append(f'class: {loc.strip()}.__init__')
                current_class_name = loc
                
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
                
                relevant_class = get_class_by_name(class_name, pred_file, classes)
                if len(relevant_class) == 0:
                    logging.info(f"{class_name} class could not be found")
                    # handle cases like "function: my_method.inner_method"
                    loc = loc.split('.')[0]
                else:
                    relevant_method = [
                        method
                        for method in relevant_class[0]["methods"]
                        if method["name"] == method_name
                    ]
                    if len(relevant_method) == 0:
                        logging.info(f"{full_loc} method could not be found")
                    else:
                        # if method_name == '__init__' and f'{pred_file}:{class_name}' not in found_edit_modules:
                        if method_name == '__init__':
                            found_edit_modules.append(f'{pred_file}:{class_name}')
                        else:
                            # _module_name = f'function: {class_name}.{method_name}'
                            _module_name = f'{pred_file}:{class_name}.{method_name}'
                            # if _module_name not in found_edit_modules:
                            found_edit_modules.append(_module_name)
                        
                    continue
                
            # 直接搜索是否存在该function
            relevant_function = get_function_by_name(loc, pred_file, functions=functions)
            
            # 没有找到该function
            if not relevant_function or len(relevant_function) == 0:
                logging.info(f"{loc} function could not be found")
                
                if current_class_name != "":
                    # check if its a method
                    relevant_class = get_class_by_name(current_class_name, pred_file, classes)
                    # print(pred_file, current_class_name, relevant_class)
                    relevant_method = get_function_by_name(loc, pred_file, cur_class=relevant_class)
                    if len(relevant_method) == 0:
                        logging.info(f"{loc} method could not be found")
                    else:
                        # if loc == '__init__' and f'{pred_file}:{current_class_name}' not in found_edit_modules:
                        if loc == '__init__':
                            found_edit_modules.append(f'{pred_file}:{current_class_name}')
                        else:
                            _module_name = f'{pred_file}:{current_class_name}.{loc}'
                            # if _module_name not in found_edit_modules:
                            found_edit_modules.append(_module_name)
                        
                else:
                    # look for it in any class
                    cls_n, relevant_method = get_function_by_name(loc, pred_file, classes=classes)
                    if len(relevant_method) == 1:
                        # _module_name = f'function: {cls_n}.{loc}'
                        _module_name = f'{pred_file}:{cls_n}.{loc}'
                        # if _module_name not in found_edit_modules:
                        found_edit_modules.append(_module_name)
            # elif f'{pred_file}:{loc}' not in found_edit_modules:
            else:
                found_edit_modules.append(f'{pred_file}:{loc}')
        # - end identify function -
        
        elif not ignore_variable and loc.startswith(("variable:", "variables:")):
            vars = loc.split(':')[-1].strip().replace(',', ' ').split()
            # print(vars)
            for v in vars:
                if global_vars and v in global_vars:
                    # if f'variable: {v}' not in found_edit_modules:
                    # if f'{pred_file}:{v}' not in found_edit_modules:
                    found_edit_modules.append(f'{pred_file}:{v}')
        else:
            if loc.strip():
                logging.info(f"loc {loc} not recognised")

    loc_weights = collections.defaultdict(float)
    # Apply the selected merging method
    if ranking_method == "majority":
        # Majority Voting: Count the frequency of each edit location
        loc_counts = Counter(found_edit_modules)
        for loc, count in loc_counts.items():
            loc_weights[loc] = count
    elif ranking_method == "mrr":
        sample_locs = []
        for loc in found_edit_modules:     
            # Calculate MRR for edit locations
            for rank, loc in enumerate(sample_locs, start=1):
                loc_weights[loc] += 1 / rank
            
    # Sort edit locations based on weight
    ranked_loc_weights = sorted(loc_weights.items(), key=lambda x: x[1], reverse=True)
    # print(ranked_loc_weights)
    res_edit_modules = [loc for loc, _ in ranked_loc_weights]
    # found_edit_module_loc = [['\n'.join(modules)] for modules in found_edit_modules]
    return res_edit_modules


def get_modules_from_line_numbers(line_numbers, pred_file, structure, 
                                  global_vars: dict=None,
                                  keep_whole_class: bool = False,
                                  keep_line: bool = False,
                                  ignore_variable: bool = False,
                                  ):
    found_modules = []
    cur_module_end_line = None
    for line in line_numbers:
        # check if global var
        if not ignore_variable and global_vars:
            variable = is_global_var(line, global_vars)
            if variable:
                found_modules.append(f"variable: {variable}")
                continue
        if cur_module_end_line and line <= cur_module_end_line:
            continue
        module, cur_module_end_line = get_module_from_line_number(line, pred_file, structure)
        if not module:
            if keep_line:
                found_modules.append(f"line: {line}")
            continue
        
        if module not in found_modules:
            found_modules.append(module)
        if keep_whole_class and '.' in module:
            module = module.split(':')[1].strip()
            class_name, _ = module.split('.')
            if f'class: {class_name}' not in found_modules:
                found_modules.append(f'class: {class_name}')
    return found_modules


def get_module_from_line_number(line, file_path, structure):
    path = file_path.split('/')
    s = deepcopy(structure)   # stuck here
    for p in path:
        if p in s:
            s = s[p]
        else:
            return (None, None)
    
    for txt in s['classes']:
        for func in txt['methods']:
            if line >= func['start_line'] and line <= func['end_line']:
                if func['name'] == '__init__':
                    desc = f"class: {txt['name']}"
                    return (desc, None)
                else:
                    desc = f"function: {txt['name']}.{func['name']}"
                    return (desc, func['end_line'])
        
        # if there is no methods in this class, return class
        # if not txt['methods'] and line >= txt['start_line'] and line <= txt['end_line']:
        #     desc = f"class: {txt['name']}"
        #     return (desc, None)
        
        # if not in any function but in class content, return class
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"class: {txt['name']}"
            return (desc, None)
        
    for txt in s['functions']:
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"function: {txt['name']}"
            return (desc, txt['end_line'])
        
    return (None, None)