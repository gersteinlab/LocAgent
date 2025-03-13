from agentless.util.utils import load_jsonl
from utils.process_output import (
    parse_keyword_json_obj,
)
import json
from tqdm import tqdm
from datasets import load_dataset
import litellm
import os
from utils.util import convert_to_json
from graph_encoder.dependency_graph import RepoSearcher, RepoEntitySearcher
import pickle
from plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
    search_entity_in_global_dict,
    # get_current_issue_data,
    get_current_repo_modules,
    find_matching_files_from_list,
    get_module_name_by_line_num
)
from graph_encoder.dependency_graph.build_graph_v2 import (
    NODE_TYPE_CLASS, NODE_TYPE_FUNCTION,
)

import networkx as nx

def mean_shortest_distance(graph, list_a, list_b):
    # Step 1: Compute shortest distances
    mean_distances = []
    
    for b in list_b:
        shortest_dist_to_a = float('inf')  # Initialize with a very large number
        for a in list_a:
            try:
                # Calculate the shortest path length
                dist = nx.shortest_path_length(graph, source=a, target=b)
                shortest_dist_to_a = min(shortest_dist_to_a, dist)
            except nx.NetworkXNoPath:
                # Handle cases where there is no path
                continue
        if shortest_dist_to_a != float('inf'):  # Valid distance found
            mean_distances.append(shortest_dist_to_a)
    
    # Step 3: Calculate the mean value
    return sum(mean_distances) / len(mean_distances) if mean_distances else None


PR_TEMPLATE="""
Given the following GitHub problem description, classify the problem statement into the following categories: Problem description, error trace, code to reproduce the bug, and additional context.
--- BEGIN PROBLEM STATEMENT ---

{problem_statement}

--- END PROBLEM STATEMENT ---
"""

EXTRACT_TASK="""Then identify all the potential modules in the '{repo_name}' package mentioned by each category.
Your output should consist of a list of JSON objects, each containing:  
- **keyword**: The class name or function name mentioned.
- **possible_file_path**: A possible file path in the repository where the module might be located.  
- **possible_line_numbers**: An array of line numbers where the module is likely relevant (if applicable or identifiable).  

Example output:  
```json
[
    {{"keyword": "func_1", "possible_file_path": "path/to/file.py", "possible_line_numbers": [10, 25]}},
    {{"keyword": "class_A", "possible_file_path": "path/to/file2.py", "possible_line_numbers": []}},
    {{"keyword": "class_B.func_2", "possible_file_path": "path/to/file3.py", "possible_line_numbers": []}}
]
"""

def extract_keywords(problem_statement, repo_name):
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant that can interact with a computer to solve tasks.\n<IMPORTANT>\n* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.\n</IMPORTANT>\n"
        },
        {"role": "system", 
        "content": PR_TEMPLATE.format(problem_statement=problem_statement)
        }
    ]
    response = litellm.completion(
                    model='azure/gpt-4o',
                    messages=messages
                )
    messages.append(convert_to_json(response.choices[0].message))
    messages.append({
        'role': 'user',
        'content': EXTRACT_TASK.format(repo_name=repo_name)
    })
    response = litellm.completion(
                    model='azure/gpt-4o',
                    messages=messages
                )
    messages.append(convert_to_json(response.choices[0].message))
    identified_str = response.choices[0].message.content
    parsed_search_terms = parse_keyword_json_obj(identified_str)
    return parsed_search_terms, messages

gt_file = os.environ.get("GT_MODULES_FILE")
gt_data = load_jsonl(gt_file)
gt_data_dict = {}
for data in gt_data:
    instance_id = data['instance_id']
    gt_data_dict[instance_id] = data

dataset_name = 'princeton-nlp/SWE-bench_Lite'
split = 'test'
swe_bench_data = load_dataset(dataset_name, split=split)
output_folder = 'outputs_data/gpt-4o/extract_keywords'
output_file = 'extract_keywords_outputs.jsonl'
output_file = os.path.join(output_folder, output_file)
processed_instance = []
if os.path.exists(output_file):
    processed_data = load_jsonl(output_file)
    processed_instance = [data['instance_id'] for data in processed_data]

swe_bench_data = swe_bench_data.select(range(0, 300))
for instance in tqdm(swe_bench_data):
    if instance['instance_id'] in processed_instance:
        continue
    
    search_terms, msg_history = extract_keywords(instance['problem_statement'], instance['instance_id'].split('_')[0])
    extracted_data = {
        "instance_id": instance['instance_id'],
        'search_terms': search_terms,
        'file_changes': gt_data_dict[instance['instance_id']]['file_changes'],
        'problem_statement': instance['problem_statement'],
        'patch': instance['patch'],
        'repo': instance['repo'],
        'base_commit': instance['base_commit'],
        'messages': msg_history,
        
    }
    with open(output_file, 'a') as f:
        f.write(
            json.dumps(extracted_data) + "\n"
        )
    # break
    

# output_path = '/home/gangda/workspace/czl/swebench/outputs_data/gpt-4o/extract_keywords/extract_keywords_outputs.jsonl'
output_data = load_jsonl(output_file)
# identified_output_file = '/home/gangda/workspace/czl/swebench/outputs_data/gpt-4o/extract_keywords/identified_keywords_outputs.jsonl'
identified_output_file = os.path.join(output_folder, 'identified_keywords_outputs.jsonl')
processed_instance = []
if os.path.exists(identified_output_file):
    processed_data = load_jsonl(identified_output_file)
    processed_instance = [data['instance_id'] for data in processed_data]
    
for output in tqdm(output_data):
    if output['instance_id'] in processed_instance:
        continue
    
    gt_entities = []
    for module in output['file_changes'][0]['changes']['edited_modules']:
        if module.startswith('variable:'):
            gt_entities.append(output['file_changes'][0]['file'])
        else:
            gt_entities.append(output['file_changes'][0]['file']+':'+module.split(': ')[-1].strip())
            
    instance_id = output['instance_id']
    set_current_issue(instance_data=output)
    DEPENDENCY_GRAPH_LOC = os.environ.get("DEPENDENCY_GRAPH_LOC")
    G = pickle.load(
        open(f"{DEPENDENCY_GRAPH_LOC}/{instance_id}.pkl", "rb")
    )
    entity_searcher = RepoEntitySearcher(G)
    identified_entities = []
    identified_files = []
    for search_term in output['search_terms']:
        keyword = search_term['keyword']
        possible_file_path = search_term['possible_file_path']
        possible_line_numbers = search_term['possible_line_numbers']
        if possible_file_path and entity_searcher.has_node(possible_file_path):
            identified_entities.append(possible_file_path)
            identified_files.append(possible_file_path)
        
        term = f'{possible_file_path}:{keyword}'
        if entity_searcher.has_node(term):
            identified_entities.append(term)
            continue
        elif term.endswith('.__init__'):
            nid = term[:-(len('.__init__'))]
            if entity_searcher.has_node(nid):
                identified_entities.append(nid)
                continue
        
        files, classes, _ = get_current_repo_modules()
        all_file_paths = [file[0] for file in files]
        include_files = all_file_paths
        file_pattern = possible_file_path if possible_file_path else ''
        if file_pattern:
            include_files = find_matching_files_from_list(all_file_paths, file_pattern)
        if not include_files:
            include_files = all_file_paths
        
        found_entities_dict = search_entity_in_global_dict(term, include_files)
        if not found_entities_dict:
            found_entities_dict = search_entity_in_global_dict(term)
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
            
        if found_entities_dict:
            for ntype, nids in found_entities_dict.items():
                if not nids: continue
                # class 和 function 逻辑一致(3个以内显示)
                if ntype in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS]:
                    identified_entities.extend(nids)
        if possible_line_numbers:
            for line_num in possible_line_numbers:
                module = get_module_name_by_line_num(possible_file_path, line_num)
                if module:
                    identified_entities.append(module['node_id'])
    
    output['gt_entities'] = list(set(gt_entities))
    output['identified_files'] = list(set(identified_files))
    output['identified_entities'] = list(set(identified_entities))
    output['num_of_entities'] = len(list(set(identified_entities)))
    mean_dist = mean_shortest_distance(G, output['identified_entities'], output['gt_entities'])
    output['mean_dist'] = mean_dist
    with open(identified_output_file, 'a') as fa:
        fa.write(
            json.dumps(output) + "\n"
        )
    reset_current_issue()
    
    