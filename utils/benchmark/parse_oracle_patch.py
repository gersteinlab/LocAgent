import os
import json
import ast
import re
import subprocess
import tempfile
import collections
import unidiff
import argparse
import logging
import logging.handlers
from datetime import datetime
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
import tokenize
from io import StringIO
from utils.benchmark.setup_swebench_repo import (
    setup_swebench_repo, 
    setup_swebench_lite_repo
)
from utils.benchmark.get_patch_info import get_oracle_filenames
from plugins.location_tools.utils.get_repo_structure.get_repo_structure import (
    parse_python_file,
)
from agentless.util.parse_global_var import parse_global_var_from_code
import torch.multiprocessing as mp
from queue import Empty
import uuid
import shutil
# from utils.get_patch_info import *

# testcase
# code = "    from django.core.management.color import color_style "
# code = "import sys"
# def is_import_statement(code_str):
#     try:
#         parsed = ast.parse(code_str.strip())
#         for node in ast.walk(parsed):
#             if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
#                 return True
#     except (ValueError, SyntaxError) as e:
#         return False
#     return False

def is_import_statement(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False


def is_comment(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False 


def is_docstring(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False 


def get_import_nodes(target_file):
    with open(target_file, 'r') as f:
        source_code = f.read()

    # Parse the source code
    tree = ast.parse(source_code)
    class ImportCollector(ast.NodeVisitor):
        def __init__(self):
            self.imports = []

        def visit_Import(self, node):
            self.imports.append({
                "type": "import",
                "module": None,  # Regular imports don't specify a module
                "names": [alias.name for alias in node.names],
                "start_line": node.lineno,
                "end_line": getattr(node, 'end_lineno', node.lineno)  # Use node.lineno if end_lineno is not available
            })
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            self.imports.append({
                "type": "from import",
                "module": node.module,
                "names": [alias.name for alias in node.names],
                "start_line": node.lineno,
                "end_line": getattr(node, 'end_lineno', node.lineno)  # Use node.lineno if end_lineno is not available
            })
            self.generic_visit(node)

    import_collector = ImportCollector()
    import_collector.visit(tree)

    # return the collected imports
    return import_collector.imports

    
def get_comment_nodes(target_file):
    comment_nodes = []
    with open(target_file, 'r') as f:
        source_code = f.read()
    # Tokenize the source code to find comments and their locations
    source = StringIO(source_code)
    tokens = tokenize.generate_tokens(source.readline)

    for token_type, token_string, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            # For comments, this will usually be the same as start_line
            comment_nodes.append({
                "start_line": start[0],
                "end_line": end[0],
                "content": token_string
            })
            logging.debug(f"Found comment: {token_string} starting at line {start[0]} and ending at line {end[0]}")
    return comment_nodes


def parse_class_docstrings(target_file: str) -> list:
    with open(target_file, 'r') as f:
        source_code = f.read()
        
    # Parse the code string
    parsed_code = ast.parse(source_code)
    docstring_nodes = []
    # Iterate through nodes to find the class definition
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.ClassDef):
            # Retrieve the class docstring
            docstring = ast.get_docstring(node)
            if docstring:
                # Find the start and end lines of the docstring
                start_line = node.body[0].lineno  # First node under the class is usually the docstring
                end_line = start_line + len(docstring.splitlines()) - 1
                docstring_nodes.append({
                    'start_line': start_line,
                    'end_line': end_line,
                    'content': docstring
                })
    return docstring_nodes


def parse_module_name(code_str: str):
    # Regular expression to match the function definition and extract the name
    match = re.search(r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_str)

    if match:
        function_name = match.group(1)
        return function_name
    else:
        # print("No function definition found.")
        return None
    
def parse_patch(patch, ignore_import=True):
    """
    Parse a git patch into a structured format.

    Parameters:
        patch (str): The git patch as a string.

    Returns:
        list: A list of dictionaries representing the file changes.
    """
    parsed_patches = []
    patch_set = unidiff.PatchSet(patch)
    # Iterate over each file in the patch set
    for patched_file in patch_set:
        if not str(patched_file.path).endswith('.py'):
            continue
        parsed_file_patch = dict()
        parsed_file_patch['file'] = patched_file.path
        parsed_file_patch['hunks'] = []
        # logging.debug(f"File: {patched_file.path}")
        # Iterate over each hunk (a block of changes) in the file
        for hunk in patched_file:
            parsed_hunk = {
                'start_line': hunk.source_start,
                # 'edited_modules': [],
                # 'added_modules': [],
                'changes': defaultdict(list)
            }
            # print(f"  Hunk starting at line {hunk.source_start}")
            
            # print("==========")
            # print(hunk.section_header)
            # current_module = None
            # if 'def ' in hunk.section_header:
            #     current_module = parse_module_name(hunk.section_header)
            #     # print(current_module)
            # for line in hunk.source:
            #     if 'def ' in line:
            #         current_module = parse_module_name(line)
            #     elif line.startswith('-'):
            #         if current_module and current_module not in parsed_hunk['edited_modules']:
            #             parsed_hunk['edited_modules'].append(current_module)
                        
            # current_module = None
            # if 'def ' in hunk.section_header:
            #     current_module = parse_module_name(hunk.section_header)
            # for line in hunk.target:
            #     sign_line = False
            #     if 'def ' in line:
            #         current_module = parse_module_name(line)
            #         sign_line = True # this line is a signature
            #     elif line.startswith('+'):
            #         if current_module and current_module not in parsed_hunk['edited_modules']:
            #             if sign_line:
            #                 parsed_hunk['added_modules'].append(current_module)
            #             else:
            #                 parsed_hunk['edited_modules'].append(current_module)
            
            # Iterate over each line in the hunk
            for line in hunk:
                if not str(line)[1:].strip():
                    continue
                
                if line.is_removed:
                    # code_line = str(line)[1:].strip()
                    # if not is_import_statement(code_line):
                    parsed_hunk['changes']['delete'].append({
                                # "type": change_type,
                                "content": str(line)[1:],
                                "line": line.source_line_no,
                            })
                    
                elif line.is_added:
                    # code_line = str(line)[1:].strip()
                    # print(code_line)
                    # if not is_import_statement(code_line): # and code_line # ignore adding space?
                    parsed_hunk['changes']['add'].append({
                                # "type": change_type,
                                "content": str(line)[1:],
                                "line": line.target_line_no,
                            })
            parsed_file_patch['hunks'].append(parsed_hunk)

        parsed_patches.append(parsed_file_patch)
    return parsed_patches


def check_moduel_existed(module, file_structure):
    s = file_structure
    module_type = module.split(':')[0].strip()
    module_name = module.split(':')[-1].strip()
    
    if module_type == 'function' and '.' not in module_name:
        for func in s['functions']:
            if func['name'] == module_name:
                return True
    elif module_type == 'function' and '.' in module_name:
        class_name = module_name.split('.')[0]
        method_name = module_name.split('.')[-1]
        cls = [cls for cls in s['classes'] if cls['name'] == class_name]
        if cls:
            method = [method for method in cls[0]['methods'] if method['name'] == method_name]
            if method:
                return True
    elif module_type == 'class':
        cls = [cls for cls in s['classes'] if cls['name'] == module_name]
        if cls:
            return True
        
    return False


# def get_module_from_line_number_with_file_structure(line, file_structure, include_class=False, merge_init=True):
def get_module_from_line_number_with_file_structure(line, file_structure, include_class=False, merge_init=False):
    s = file_structure
    for txt in s['classes']:
        for func in txt['methods']:
            if line >= func['start_line'] and line <= func['end_line']:
                if merge_init and func['name'] == '__init__':
                    desc = f"class: {txt['name']}"
                    return desc
                else:
                    desc = f"function: {txt['name']}.{func['name']}"
                    return desc
            
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"class: {txt['name']}"
            if not txt['methods'] or include_class:
                return desc
            
    for txt in s['functions']:
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"function: {txt['name']}"
            return desc
        
    return None


def apply_patch_str(patch, apply_file_path, hunk_size):
    # Write the patch string to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_patch_file:
        temp_patch_file.write(patch)
        temp_patch_file_path = temp_patch_file.name

    # Apply the patch using the `patch` utility
    try:
        result = subprocess.run(
            ['patch', '-p1', '-i', temp_patch_file_path, apply_file_path],
            check=True,
            text=True,
            capture_output=True
        )
        # print("Patch applied successfully.")
        logging.debug(result.stdout)
        offsets = [0 for i in range(hunk_size)]
        for out in str(result.stdout).splitlines():
            # if out.startswith('patching file'):
            #     offsets.append(0)
            # else:
                # process offset
                # Regular expression to extract offset (including negative values)
            pattern = r"Hunk #(\d+) succeeded at (\d+) \(offset ([+-]?\d+) lines\)"
            match = re.search(pattern, str(out))
            # Extracting the values if a match is found
            if match:
                hunk_id = int(match.group(1))
                offset = int(match.group(3))
                offsets[hunk_id-1] = offset
            # else:
            #     offset = 0
            # for i in range(hunk_id):
            #     if i == (hunk_id-1): offsets.append(offset)
            #     if i >= len(offsets): offsets.append(0) # padding
                
        logging.debug('offsets', offsets)
        return (True, offsets)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error applying patch: {e.stderr}")
        return (False, [])
    finally:
        # Clean up the temporary file
        import os
        os.remove(temp_patch_file_path)


def map_import_lines(codes):
    in_import_statement = False
    open_parens = 0
    line_labels = {}  # Dictionary to store line number and its label (True/False)
    for code in codes:
        content = code['content']
        line_num = code['line']
        stripped_line = content.strip()
        if not in_import_statement:
            if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                in_import_statement = True
                open_parens += stripped_line.count('(') - stripped_line.count(')')
                line_labels[line_num] = True
                if open_parens == 0:
                    in_import_statement = False
            else:
                # Not an import statement
                line_labels[line_num] = False
        else:
            # Inside a multi-line import statement
            open_parens += stripped_line.count('(') - stripped_line.count(')')
            line_labels[line_num] = True
            if open_parens == 0:
                in_import_statement = False
    return line_labels


def parse_global_var_from_file(file_path):
    with open(file_path, 'r') as f:
        file_content = f.read()
    global_vars = parse_global_var_from_code(file_content)
    return global_vars


def is_global_var(line, global_vars):
    for gvar, lrange in global_vars.items():
        if line >= lrange['start_line'] and line <= lrange['end_line']:
            return gvar
    return None


def group_patch_by_file(patch):
    """
    Groups a patch string by file.

    Args:
        patch (str): The patch content as a string.

    Returns:
        dict: A dictionary where the keys are file paths, and the values are the corresponding patch content.
    """
    patch_by_file = defaultdict(list)
    patch_lines = patch.splitlines()

    current_file = None
    current_hunks = []
    file_header_pattern = r"^(---|\+\+\+) (.+)"

    for line in patch_lines:
        match = re.match(file_header_pattern, line)
        if match:
            current_file = re.sub(r"^(a/|b/)", "", match.group(2))
            patch_by_file[current_file].append(f"{line}\n")
            # if line.startswith('---'):
                
            # elif line.startswith('+++') and current_file:
            #     patch_by_file[current_file].append(f"{line}\n")
        else:
            if current_file:
                patch_by_file[current_file].append(f"{line}\n")

    return {file: "".join(hunks) for file, hunks in patch_by_file.items()}


def extract_module_from_patch(instance, repo_dir, max_edit_file_num=1, rank=0):
    # instance_id = instance['instance_id']
    # print(instance_id)
    
    # instance = [bug for bug in swe_bench_test_data if bug['instance_id']==instance_id][0]
    edit_files = get_oracle_filenames(instance['patch'])
    # print(len(edit_files))
    filtered_edit_files = []
    for fle in edit_files:
        if fle.endswith('.py'):
            filtered_edit_files.append(fle)
    if not len(filtered_edit_files):
        return None
    if len(filtered_edit_files) > max_edit_file_num:
        return None
    # pass_flag = False
    # for file in filtered_edit_files:
    #     if not file.endswith('.py'):
    #         pass_flag = True
    # if pass_flag:
    #     return None
    # logging.debug(f"--- {rank} parse_patch success ----")
    
    file_changes = parse_patch(instance['patch'])
    
    # print(rank, instance['instance_id'], len(file_changes))
    # Group the patch by file
    patch_by_file = group_patch_by_file(instance['patch'])
    
    updated_file_changes = []
    for file_change in file_changes:
        file = file_change['file']
        # logging.debug(f"--- {rank} check start----")
        # logging.debug(f"{rank}: len(file_changes) {len(file_changes)}")
        # logging.debug(f"{rank}: file_change.keys, {file_change.keys()}")
        # logging.debug(f"--- {rank} check end----")
        
        if not file.endswith('.py'): continue
        
        target_file_path = os.path.join(repo_dir, file)
        # print(target_file_path)
        
        # initial file structure
        class_info, function_names, file_lines = parse_python_file(target_file_path)
        old_file_structure = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }
        old_global_vars = parse_global_var_from_file(target_file_path)
        old_import_nodes = get_import_nodes(target_file_path)
        old_comment_nodes = get_comment_nodes(target_file_path)
        old_docstring_nodes = parse_class_docstrings(target_file_path)
        
        # Extract the partial patch for this file
        partial_patch = patch_by_file.get(file)
        if not partial_patch:
            logging.warning(f"No patch found for {file}")
            continue
        
        # Apply the patch
        success, offsets = apply_patch_str(partial_patch, target_file_path, len(file_change['hunks']))
        if not success:
            # TODO: assert
            return
        
        # new file structure
        class_info, function_names, file_lines = parse_python_file(target_file_path)
        new_file_structure = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }
        new_global_vars = parse_global_var_from_file(target_file_path)
        new_import_nodes = get_import_nodes(target_file_path)
        new_comment_nodes = get_comment_nodes(target_file_path)
        new_docstring_nodes = parse_class_docstrings(target_file_path)
        
        changes = collections.defaultdict(list)
        for i, hunk in enumerate(file_change['hunks']):
            # if i == len(offsets): offsets.append(0) # align with hunk size
            
            # process edited lines
            delete_change = hunk['changes']['delete']
            add_change = hunk['changes']['add']
            deleted_lines, added_lines = [], []
            
            for delete in delete_change:
                line = delete['line'] + offsets[i]
                # is_comment(line, old_comment_nodes) or \
                if is_import_statement(line, old_import_nodes) or \
                    delete['content'].strip().startswith('#') or \
                    is_docstring(line, old_docstring_nodes):
                    continue
                
                # check is global var
                variable = is_global_var(line, old_global_vars)
                if variable and variable not in changes['edited_modules']:
                    changes['edited_modules'].append(f'variable: {variable}')
                    continue
                
                # check is module
                module = get_module_from_line_number_with_file_structure(line, old_file_structure)
                if not module and delete['content'].strip():
                    deleted_lines.append(delete)
                elif module and not module in changes['edited_modules']:
                    changes['edited_modules'].append(module)
                    
            for add in add_change:
                # is_comment(line, new_comment_nodes) or \
                line = add['line'] + offsets[i]
                if is_import_statement(line, new_import_nodes) or \
                    add['content'].strip().startswith('#') or \
                    is_docstring(line, new_docstring_nodes):
                    # print(is_import_statement(line, new_import_nodes))
                    # print(is_comment(line, new_comment_nodes))
                    # print(is_docstring(line, new_docstring_nodes))
                    continue
                
                # check is global var
                variable = is_global_var(line, new_global_vars)
                if variable:
                    if variable in old_global_vars and f'variable: {variable}' not in changes['edited_modules']:
                        changes['edited_modules'].append(f'variable: {variable}')
                    elif variable not in old_global_vars and f'variable: {variable}' not in changes['added_modules']:
                        changes['added_modules'].append(f'variable: {variable}')
                    continue
                
                # check is module
                module = get_module_from_line_number_with_file_structure(line, new_file_structure)                
                if not module and add['content'].strip():
                    is_class = get_module_from_line_number_with_file_structure(line, new_file_structure, include_class=True)
                    if is_class and add['content'].strip().startswith('class'):
                        changes['added_modules'].append(is_class)
                    else:
                        # pdb.set_trace()
                        # added_lines.append(f'line: {line}')
                        # print(add)
                        added_lines.append(add)
                elif module and \
                    module not in changes['edited_modules'] and \
                    module not in changes['added_modules']:
                    # if '.' in module:
                    #     module_cls = module.split(':')[-1].strip().split('.')[0]
                    # check if the module in old file
                    if check_moduel_existed(module, old_file_structure):
                        changes['edited_modules'].append(module)
                    else:
                        changes['added_modules'].append(module)
                        # if '.' in module:
                        #     cls_n = module.split(':')[-1].split('.')[0].strip()
                        #     if check_moduel_existed(f'class: {cls_n}', old_file_structure) and \
                        #         f'class: {cls_n}' not in changes['edited_modules']:
                        #         changes['edited_modules'].append(f'class: {cls_n}')
            
            # join the lines to find import statement
            # deleted_lines_import_labels = map_import_lines(deleted_lines)
            # added_lines_import_labels = map_import_lines(added_lines)
            
            for delete in deleted_lines:
                line = delete['line']
                # if deleted_lines_import_labels[line]:
                #     continue
                module = get_module_from_line_number_with_file_structure(line, old_file_structure, True)
                if module and module not in changes['edited_modules']:
                        changes['edited_modules'].append(module)
                        continue
                if f'line: {line}' not in changes['edited_lines']:
                    changes['edited_lines'].append(f'line: {line}')
                    
            for add in added_lines:
                line = add['line']
                
                # if added_lines_import_labels[line]:
                #     continue
                module = get_module_from_line_number_with_file_structure(line, new_file_structure, True)
                
                if module:
                    if module not in changes['edited_modules'] and check_moduel_existed(module, old_file_structure):
                        changes['edited_modules'].append(module)
                        continue
                    if module not in changes['added_modules']:
                        changes['added_modules'].append(module)
                elif f'line: {line}' not in changes['edited_lines']:
                    changes['added_lines'].append(f'line: {line}')
        
        _changes = collections.defaultdict(list)
        for mode, change in changes.items():
            if mode in ['added_lines', 'edited_lines']:
                continue
            for c in change:
                if c.startswith("variable:"):
                    continue
                if mode in ['added_modules', 'edited_modules']:
                    _mode = mode.replace('_modules', '_entities')
                    _changes[_mode].append(f'{file}:{c.split(':')[-1].strip()}')
                
                if c.startswith("function:") and '.' in c:
                    _c = c.split(':')[-1].strip().split('.')[0]
                    if f'{file}:{_c.strip()}' not in _changes[mode]:
                        _changes[mode].append(f'{file}:{_c.strip()}')
                else:
                    if f'{file}:{c.split(':')[-1].strip()}' not in _changes[mode]:
                        _changes[mode].append(f'{file}:{c.split(':')[-1].strip()}')
        
        updated_file_changes.append({
            'file': file,
            # 'changes': changes
            'changes': _changes
        })
    
    return updated_file_changes


def generate_oracle_modules_for_lite(repo_base_dir='playground'):
    swe_bench_test_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file = f'evaluation/gt_data/SWE-bench_Lite/gt_modules_data_{current_date}.jsonl'
    processed_instances = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                processed_instances.append(json.loads(line)['instance_id'])
                
    for instance in tqdm(swe_bench_test_data):
        if instance['instance_id'] in processed_instances:
            continue
        
        # pull the repo
        os.makedirs(repo_base_dir, exist_ok=True)
        repo_dir = setup_swebench_lite_repo(instance_data=instance, repo_base_dir=repo_base_dir)
    
        file_changes = extract_module_from_patch(instance, repo_dir)
        with open(output_file, 'a') as f:
            f.write(json.dumps({
                'instance_id': instance['instance_id'],
                'file_changes': file_changes,
                'repo': instance['repo'],
                'base_commit': instance['base_commit'],
                'problem_statement': instance['problem_statement'],
                'patch': instance['patch']
            }) + '\n')
    
    return output_file


def generate_oracle_modules_for_train(repo_base_dir='playground/SWE-bench', selected_list=None):
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench", split="train")
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file = f'evaluation/gt_data/SWE-bench/gt_modules_data_{current_date}.jsonl'
    processed_instances = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                processed_instances.append(json.loads(line)['instance_id'])
                
    error_list, empty_edit_list = [], []
    for instance in tqdm(swe_bench_data):
        if instance['instance_id'] in processed_instances:
            continue
        if selected_list and instance['instance_id'] not in selected_list:
            continue
        
        
        try:
            # pull the repo
            os.makedirs(repo_base_dir, exist_ok=True)
            repo_dir = setup_swebench_repo(instance_data=instance, repo_base_dir=repo_base_dir)
            
            file_changes = extract_module_from_patch(instance, repo_dir)
            if not file_changes:
                empty_edit_list.append(instance['instance_id'])
                continue
            else:
                for fchange in file_changes:
                    if not fchange['changes']: # or \
                        # "edited_modules" not in fchange['changes'] or \
                        # not fchange['changes']["edited_modules"]:
                        empty_edit_list.append(instance['instance_id'])
                        continue
            
            with open(output_file, 'a') as f:
                f.write(json.dumps({
                    'instance_id': instance['instance_id'],
                    'file_changes': file_changes,
                    'repo': instance['repo'],
                    'base_commit': instance['base_commit'],
                    'problem_statement': instance['problem_statement'],
                    'patch': instance['patch']
                }) + '\n')
        except FileNotFoundError:
            error_list.append(instance['instance_id'])
    print(error_list)
    # ['apache__airflow-6783', 'jupyterlab__jupyterlab-14038']
    print(empty_edit_list)
    # ['Qiskit__qiskit-10347', 'Qiskit__qiskit-10555', 'Qiskit__qiskit-6700', 'apache__airflow-9759', 'google__jax-1880', 'pyca__cryptography-1988']
    return output_file


def run_extract_module_from_patch(rank, 
                                  queue, log_queue, output_file_lock,
                                  repo_playground, output_file, max_edit_file_num
                                  ):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName("DEBUG"))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")
    
    while True:
        try:
            instance = queue.get_nowait()
        except Empty:
            break
        
        try:
            # pull the repo
            repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))
            os.makedirs(repo_playground, exist_ok=True)
            repo_dir = setup_swebench_repo(instance_data=instance, repo_base_dir=repo_playground)
            file_changes = extract_module_from_patch(instance, repo_dir, max_edit_file_num=max_edit_file_num, rank=rank)
            if not file_changes:
                # empty_edit_list.append(instance['instance_id'])
                continue
            else:
                for fchange in file_changes:
                    if not fchange['changes']: # or \
                        # "edited_modules" not in fchange['changes'] or \
                        # not fchange['changes']["edited_modules"]:
                        # empty_edit_list.append(instance['instance_id'])
                        continue
            with output_file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps({
                        'instance_id': instance['instance_id'],
                        'file_changes': file_changes,
                        'repo': instance['repo'],
                        'base_commit': instance['base_commit'],
                        'problem_statement': instance['problem_statement'],
                        'patch': instance['patch']
                    }) + '\n')
        except FileNotFoundError:
            logger.debug(f"rank {rank}: FileNotFoundError.")
            
            # error_list.append(instance['instance_id'])
        except subprocess.CalledProcessError as e:
            logger.debug(f"rank {rank}: {e}")
            # error_list.append(instance['instance_id'])
        # except:
        except Exception as e:
            logger.debug(f"rank {rank}: {e}")
        finally:
            if os.path.exists(repo_playground):
                shutil.rmtree(repo_playground)
            # error_list.append(instance['instance_id'])

def generate_oracle_modules_for_loc_bench(dataset_file, gen_n_limit,
                                          max_edit_file_num=1, 
                                          repo_base_dir='playground/loc_bench',
                                          num_processes=1):
    logging.basicConfig(
        # filename=f"{args.output_folder}/localize.log",
        level=logging.getLevelName('DEBUG'),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"evaluation/gt_data/LOC-bench/gen_gt.log"),
            logging.StreamHandler()
        ]
    )
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file = f'evaluation/gt_data/LOC-bench/gt_modules_data_{max_edit_file_num}file_{current_date}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_instances = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                processed_instances.append(json.loads(line)['instance_id'])     
    
    bench_data = []
    with open(dataset_file, 'r') as f:
        for line in f:
            instance = json.loads(line)
            bench_data.append(instance)
    
    manager = mp.Manager()
    queue = manager.Queue()
    output_file_lock = manager.Lock()
    
    num_instances = 0
    for instance in bench_data[:gen_n_limit]:
        if not instance['instance_id'] in processed_instances:
            queue.put(instance)
            num_instances += 1
    
    log_queue = manager.Queue()
    queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()
    mp.spawn(
        run_extract_module_from_patch,
        nprocs=min(num_instances, num_processes) if num_processes > 0 else num_instances,
        args=(queue, log_queue, output_file_lock,
              repo_base_dir, output_file, max_edit_file_num
              ),
        join=True
    )
    queue_listener.stop()
    
    # error_list, empty_edit_list = [], []
    
    # for instance in tqdm(bench_data):
    #     if instance['instance_id'] in processed_instances:
    #         continue
    # print('error_list', error_list)
    # ['apache__airflow-6783', 'jupyterlab__jupyterlab-14038']
    # print('empty_edit_list', empty_edit_list)
    # ['Qiskit__qiskit-10347', 'Qiskit__qiskit-10555', 'Qiskit__qiskit-6700', 'apache__airflow-9759', 'google__jax-1880', 'pyca__cryptography-1988']
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_base_dir', type=str, default='playground/repo_base')
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--selected_list_file', type=str, default='playground/repo_base')
    # parser.add_argument('--merge_init', action='store_true')
    parser.add_argument('--loc_bench', action='store_true')
    parser.add_argument("--max_edit_file_num", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--gen_n_limit", type=int, default=0)
    
    args = parser.parse_args()
    # for test/debug
    # repo_base_dir = '/home/ubuntu/data/repo'
    # result = extract_module_from_patch('sympy__sympy-11870', repo_base_dir)
    # print(result)
    
    # repo_base_dir = '/home/ubuntu/data/repo'
    if args.dataset == 'princeton-nlp/SWE-bench_Lite' and args.split == 'test':
        generate_oracle_modules_for_lite(args.repo_base_dir)
    elif args.dataset == 'princeton-nlp/SWE-bench' and args.split == 'train':
        # selected_list_file = '/home/ubuntu/auto-search-agent/scripts/notebooks/fine-tune/data/selected_instances_20241022.json'
        # selected_list_file = '/home/ubuntu/auto-search-agent/scripts/notebooks/fine-tune/data/selected_instances_20241122.json'
        with open(args.selected_list_file, 'r') as f:
            selected_list = json.loads(f.read())
        
        generate_oracle_modules_for_train(args.repo_base_dir, selected_list)
        
    if args.loc_bench:
        generate_oracle_modules_for_loc_bench(args.dataset, args.gen_n_limit,
                                              args.max_edit_file_num, 
                                              args.repo_base_dir, args.num_processes)