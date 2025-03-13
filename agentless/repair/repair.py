import argparse
import copy
from copy import deepcopy
import json
import logging
import os
import re
from difflib import unified_diff
import pickle
from datasets import load_dataset
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty
from logging.handlers import QueueListener

from agentless.util.api_requests import (
    create_chatgpt_config,
    num_tokens_from_messages,
    request_chatgpt_engine,
)
from agentless.util.postprocess_data import (
    check_code_differ_by_just_empty_lines,
    check_syntax,
    extract_python_blocks,
    fake_git_repo,
    lint_code,
    parse_diff_edit_commands,
    parse_edit_commands,
    remove_empty_lines,
    split_edit_multifile_commands,
)
from agentless.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    get_repo_structure,
    line_wrap_content,
    transfer_arb_locs_to_locs,
)
from agentless.util.utils import load_jsonl
from evaluation.eval_metric import filtered_instances as FILTER_INSTANCES
import os
DEPENDENCY_GRAPH_LOC = os.environ.get("DEPENDENCY_GRAPH_LOC")
from graph_encoder.dependency_graph import RepoEntitySearcher

repair_relevant_file_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
"""
repair_relevant_file_with_scope_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
In the file below, "..." refers to some less relevant content being omited for brebity.
"""
with_scope_explanation = """
Note that "..." refers to some omited content that is not actually in the files. Your *SEARCH/REPLACE* edit must not contain such "...".
"""
repair_relevant_file_with_suspicious_loc_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs. Some suspicious locations are provided for closer inspection.
"""
repair_prompt_combine_topn = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please generate `edit_file` commands to fix the issue.

The `edit_file` command takes four arguments:

edit_file(filename: str, start: int, end: int, content: str) -> None:
    Edit a file. It replaces lines `start` through `end` (inclusive) with the given text `content` in the open file.
    Args:
    filename: str: The full file name to edit.
    start: int: The start line number. Must satisfy start >= 1.
    end: int: The end line number. Must satisfy start <= end <= number of lines in the file.
    content: str: The content to replace the lines with.

Please note that THE `edit_file` FUNCTION REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the `edit_file` command in blocks ```python...```.
"""


repair_prompt_combine_topn_cot = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate `edit_file` commands to fix the issue.

The `edit_file` command takes four arguments:

edit_file(filename: str, start: int, end: int, content: str) -> None:
    Edit a file. It replaces lines `start` through `end` (inclusive) with the given text `content` in the open file.
    Args:
    filename: str: The full file name to edit.
    start: int: The start line number. Must satisfy start >= 1.
    end: int: The end line number. Must satisfy start <= end <= number of lines in the file.
    content: str: The content to replace the lines with.

Please note that THE `edit_file` FUNCTION REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the `edit_file` command in blocks ```python...```.
"""

repair_prompt_combine_topn_cot_diff_codegraph = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

To help you better understand the contexts of the code segments, we provide a set of dependencies of the code segments. 
The dependencies reflect how the functions/classes in the code segments are referenced in the codebase. 

--- BEGIN DEPENDEICIES ---
{dependencies}
--- END DEPENDEICIES ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

repair_prompt_combine_topn_cot_diff = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

repair_prompt_combine_topn_cot_diff_with_solution = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Below is the analysis of the code segments. Additionally, several potential solutions are provided, each of which may independently resolve the issue. Please carefully analyze these solutions before adopting any of them.
--- BEGIN CODE ANALYSIS ---
{summary_content}

--- BEGIN CODE ANALYSIS ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""


repair_prompt_combine_topn_cot_diff_with_analysis = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Below is the analysis of the code segments above.
--- BEGIN CODE ANALYSIS ---
{summary_content}

--- BEGIN CODE ANALYSIS ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

REPO_GRAPH_LOC = os.environ.get("REPO_GRAPH_LOC", '/home/ubuntu/baselines/RepoGraph/repo_structures')

            
def retrieve_graph(code_graph, graph_tags, search_term, structure):
    one_hop_tags = []
    tags = []
    for tag in graph_tags:
        if tag['name'] == search_term and tag['kind'] == 'ref':
            tags.append(tag)
    for tag in tags:
        # find corresponding calling function/class
        path = tag['rel_fname'].split('/')
        s = deepcopy(structure)
        skip=False
        for p in path:
            if p in s:
                s = s[p]
            else:
                skip = True
        if skip:
            continue
        
        for txt in s['functions']:
            if tag['line'] >= txt['start_line'] and tag['line'] <= txt['end_line']:
                one_hop_tags.append((txt, tag['rel_fname']))  
        for txt in s['classes']:
            for func in txt['methods']:
                if tag['line'] >= func['start_line'] and tag['line'] <= func['end_line']:
                    func['text'].insert(0, txt['text'][0])
                    one_hop_tags.append((func, tag['rel_fname'])) 
    return one_hop_tags

def construct_code_graph_context(found_related_locs, code_graph, graph_tags, structure):
    graph_context = ""
    graph_item_format = """
### Dependencies for {func}
{dependencies}
"""
    tag_format = """
location: {fname} lines {start_line} - {end_line}
name: {name}
contents: 
{contents}

"""

    # retrieve the code graph for dependent functions and classes
    for item in found_related_locs:
        code_graph_context = ""
        item = item[0].splitlines()
        for loc in item:
            if loc.startswith("class: ") and "." not in loc:
                loc = loc[len("class: ") :].strip()
                tags = retrieve_graph(code_graph, graph_tags, loc, structure)
                for t, fname in tags:
                    code_graph_context += tag_format.format(
                        **t,
                        fname=fname,
                        contents="\n".join(t['text'])
                    )
            elif loc.startswith("function: ") and "." not in loc:
                loc = loc[len("function: ") :].strip()
                tags = retrieve_graph(code_graph, graph_tags, loc, structure)
                for t, fname in tags:
                    code_graph_context += tag_format.format(
                        **t,
                        fname=fname,
                        contents="\n".join(t['text'])
                    )
            elif "." in loc:
                loc = loc.split(".")[-1].strip()
                tags = retrieve_graph(code_graph, graph_tags, loc, structure)
                for t, fname in tags:
                    code_graph_context += tag_format.format(
                        **t,
                        fname=fname,
                        contents="\n".join(t['text'])
                    )
            else:
                continue
            graph_context += graph_item_format.format(func=loc, dependencies=code_graph_context)
    return graph_context

def _post_process_multifile_repair(
    raw_output: str,
    file_contents: dict[str, str],
    file_loc_intervals: dict[str, list],
    diff_format=False,
):
    edit_multifile_commands = extract_python_blocks(raw_output)
    edited_file = ""
    new_content = ""
    try:
        file_to_commands = split_edit_multifile_commands(
            edit_multifile_commands, diff_format=diff_format
        )
        logging.info("=== file_to_commands: ===")
        logging.info(json.dumps(file_to_commands, indent=2))
        # Let's only edit the first file in the edit commands.
        edited_file_key = next(iter(file_to_commands.keys()))
        logging.info(f"=== edited_file: {edited_file_key} ===")
        edit_commands = file_to_commands[edited_file_key]
        logging.info("=== edit_commands: ===")
        for c in edit_commands:
            logging.info(c)
            logging.info("\n" + "-" * 40)
        edited_file = eval(edited_file_key)  # convert '"file.py"' to 'file.py'
        content = file_contents[edited_file]
        if diff_format:
            new_content = parse_diff_edit_commands(
                edit_commands, content, file_loc_intervals[edited_file]
            )
        else:
            new_content = parse_edit_commands(edit_commands, content)
    except Exception as e:
        logging.error(e)
        return edited_file, new_content

    diff = list(
        unified_diff(
            content.split("\n"),
            new_content.split("\n"),
            fromfile=edited_file,
            tofile=edited_file,
            lineterm="",
        )
    )

    logging.info(f"extracted patch:")
    logging.info("\n".join(diff))
    print("\n".join(diff))
    return edited_file, new_content


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


def save_data_to_file(data: dict, file: str):
    with open(file, "a") as f:
        f.write(
            json.dumps(data) + "\n"
        )
        

def run_repair_issue(rank, args, loc_queue, log_queue, output_file_lock, swe_bench_data):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")
    
    empty_res = {
        "instance_id": "",
        "raw_output": [""],
        "try_count": [0],
        "all_generations": [[]],
        "traj": [],
        "prev_content": [[]],
        "file_names": [[]],
    }
    
    while True:
        try:
            loc = loc_queue.get_nowait()
        except Empty:
            break
        
        instance_id = loc["instance_id"]
        
        empty_res['instance_id'] = instance_id
        if len(loc["found_files"]) == 0:
            with output_file_lock:
                save_data_to_file(empty_res, args.output_file)
                logger.info(f"skipped since no files were localized")
                continue
            
        G = pickle.load(
            open(f"{DEPENDENCY_GRAPH_LOC}/{instance_id}.pkl", "rb")
        )
        searcher = RepoEntitySearcher(G)
        
        if args.loc_agent_output_format:
            pred_files, found_entities = [], []
            for entity in loc['found_entities']:      
                if not searcher.has_node(entity): continue
                entity_file = entity.split(':')[0].strip()
                if entity_file not in pred_files:
                    pred_files.append(entity_file)
                if entity not in found_entities:
                    found_entities.append(entity)
        else:
            pred_files = loc["found_files"][: args.top_n]
            # grab buggy problem issue description and structure data

        bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
        problem_statement = bench_data["problem_statement"]
        structure = get_repo_structure(
            instance_id, bench_data["repo"], bench_data["base_commit"], "playground"
        )

        files, classes, _ = get_full_file_paths_and_classes_and_functions(structure)

        raw_outputs, counts, all_generations, traj, prev_contents, file_names = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        raw_output = ""
        new_content = ""
        topn_content = ""
        # Construct file contents
        file_contents = dict()
        for i, pred_file in enumerate(pred_files):
            content = None

            for file_content in files:
                if file_content[0] == pred_file:
                    content = "\n".join(file_content[1])
                    file_contents[pred_file] = content
                    break

            assert content is not None, f"{pred_file} file not found"
            
        if args.loc_agent_output_format:
            # pred_files = []
            import collections
            file_to_edit_locs = collections.defaultdict(list)
            found_entities = found_entities[: args.top_n]
            entity_datas = searcher.get_node_data(found_entities)
            
            for entity in entity_datas:
                entity_file = entity['node_id'].split(':')[0].strip()
                entity_name = entity['node_id'].split(':')[-1].strip()
                if entity['type'] == 'function':
                    file_to_edit_locs[entity_file].append(f"function: {entity_name}")
                else:
                    cls_data = [
                        cls_ for cls_ in classes 
                        if cls_['name']==entity_name and cls_['file']==entity_file
                    ]
                    if not cls_data: continue
                    if "methods" not in cls_data[0]:
                        file_to_edit_locs[entity_file].append(f"class: {entity_name}")
                        
                    init_func = [
                        method
                        for method in cls_data[0]["methods"]
                        if method["name"] == "__init__"
                    ]
                    if init_func:
                        file_to_edit_locs[entity_file].append(f"function: {entity_name}.__init__")
                    else:
                        file_to_edit_locs[entity_file].append(f"class: {entity_name}")
                    
        else:
            # Construct top-n file context
            file_to_edit_locs = dict()
            for i, pred_file in enumerate(pred_files):
                if "found_edit_locs" in loc and len(loc["found_edit_locs"]) > i:
                    file_to_edit_locs[pred_file] = loc["found_edit_locs"][i][0].split('\n')[:5]

        topn_content, file_loc_intervals = construct_topn_file_context(
            file_to_edit_locs,
            pred_files,
            file_contents,
            structure,
            context_window=args.context_window,
            loc_interval=args.loc_interval,
            fine_grain_loc_only=args.fine_grain_loc_only,
            add_space=args.add_space,
            no_line_number=args.diff_format,
            sticky_scroll=args.sticky_scroll,
        )
        if topn_content.strip() == "":
            with output_file_lock:
                save_data_to_file(empty_res, args.output_file)
                logger.info(f"skipped since no files were localized")
            continue

        # Construct prompt.
        # Note that we assume there's no feedback, and we always use the same prompt in each turn.
        file_instruction = repair_relevant_file_instruction

        if args.cot and args.repo_graph and args.diff_format:
            depend_file_loc_intervals = dict()
            one_hop_dependencies = loc['one_hop_dependencies']
            for file_dependencies in one_hop_dependencies:
                for depend_modules in file_dependencies:
                    source_modudle = list(depend_modules.keys())[0]
                    if not depend_modules[source_modudle]:
                        continue
                    if source_modudle not in depend_file_loc_intervals:
                        depend_file_loc_intervals[source_modudle] = dict()
                    for fname, depend_modules_data in depend_modules[source_modudle].items():
                        if fname not in depend_file_loc_intervals[source_modudle]:
                            depend_file_loc_intervals[source_modudle][fname] = []
                        for m_name, line_info in depend_modules_data.items():
                            depend_file_loc_intervals[source_modudle][fname].append((line_info['start_line'], line_info['end_line']))
            
            graph_item_format = """
### Dependencies for {func}
{dependencies}
"""
            tag_format = """
location: {fname}
contents: 
{contents}

"""
            code_graph_context = ""
            if depend_file_loc_intervals:
                for source_model, depend_file_loc_interval in depend_file_loc_intervals.items():
                    dependencies_context = ""
                    # depend_file_contents = dict()
                    for f_name in depend_file_loc_interval:
                        
                        content = None
                        for file_content in files:
                            if file_content[0] == f_name:
                                content = "\n".join(file_content[1])
                                # depend_file_contents[f_name] = content
                                break

                        assert content is not None, f"{pred_file} file not found"

                        if len(depend_file_loc_interval) > 0:
                            # Note that if no location is predicted, we exclude this file.
                            file_loc_content = line_wrap_content(
                                content,
                                depend_file_loc_interval[f_name],
                                add_space=False,
                                no_line_number=True,
                                sticky_scroll=False,
                            )
                            dependencies_context += tag_format.format(
                                fname=fname,
                                contents=file_loc_content
                            )
                    code_graph_context += graph_item_format.format(
                        func=source_model,
                        dependencies=dependencies_context
                    )
                            # code_graph_context += f"### {f_name}\n{file_loc_content}\n\n\n"

                # code_graph = pickle.load(
                #     open(f"{REPO_GRAPH_LOC}/graph/{instance_id}.pkl", "rb")
                # )
                # graph_tags = json.load(
                #     open(f"{REPO_GRAPH_LOC}/tags/tags_{instance_id}.json", "r")
                # )

                # found_edit_locs = get_edit_modules(pred_files, file_to_edit_locs, structure)

                # code_graph_context = construct_code_graph_context(
                #     found_edit_locs,
                #     code_graph,
                #     graph_tags,
                #     structure,
                # )
            
            if code_graph_context:
                prompt_template = repair_prompt_combine_topn_cot_diff_codegraph
                message = prompt_template.format(
                    dependencies=code_graph_context,
                    repair_relevant_file_instruction=file_instruction,
                    problem_statement=problem_statement,
                    content=topn_content.rstrip(),  # remove trailing newlines
                ).strip()

            if not code_graph_context or num_tokens_from_messages(message, "gpt-4o-2024-05-13") > 128000:
                logger.info('messages too long!')
                prompt_template = repair_prompt_combine_topn_cot_diff
                message = prompt_template.format(
                    repair_relevant_file_instruction=file_instruction,
                    problem_statement=problem_statement,
                    content=topn_content.rstrip(),  # remove trailing newlines
                ).strip()

        elif args.cot and args.diff_format and (args.use_solution or args.use_analysis):
            if args.use_solution:
                prompt_template = repair_prompt_combine_topn_cot_diff_with_solution
            elif args.use_analysis:
                prompt_template = repair_prompt_combine_topn_cot_diff_with_analysis
            
            # summary_path='/home/ubuntu/auto-search-agent/outputs_data/results_0902/test_litellm_temp1_3/location/summary_loc_outputs_v1.jsonl'
            ins_summary = None
            with open(args.summary_file, 'r') as file:
                for line in file:
                    s = json.loads(line.strip())
                    if s["instance_id"] == instance_id:
                        ins_summary = s['thoughts']
                        break
            
            if not ins_summary:
                continue

            # file_instruction = repair_relevant_file_instruction
            message = prompt_template.format(
                repair_relevant_file_instruction=file_instruction,
                problem_statement=problem_statement,
                content=topn_content.rstrip(),  # remove trailing newlines
                summary_content="\n".join(ins_summary),
            ).strip()

        else:
            if args.cot and args.diff_format:
                prompt_template = repair_prompt_combine_topn_cot_diff
            elif args.cot:
                prompt_template = repair_prompt_combine_topn_cot
            else:
                prompt_template = repair_prompt_combine_topn        

            # file_instruction = repair_relevant_file_instruction
            message = prompt_template.format(
                repair_relevant_file_instruction=file_instruction,
                problem_statement=problem_statement,
                content=topn_content.rstrip(),  # remove trailing newlines
            ).strip()

        logger.info(f"prompting with message:\n{message}")

        sample_responses = None

        def get_response(count, non_sample_n=False):
            nonlocal sample_responses
            if count == 0:
                if args.skip_greedy:
                    return {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                if args.mock:
                    return {
                        "response": "",
                        "usage": {
                            "prompt_tokens": num_tokens_from_messages(
                                message, "gpt-4o-2024-05-13"
                            ),
                        },
                    }
                config = create_chatgpt_config(
                    message=message,
                    max_tokens=1024,
                    temperature=0,  # greedy first
                    batch_size=1,
                    model=args.model,  # use gpt-4o for now.
                )

                greedy_response = request_chatgpt_engine(config)
                return {
                    "response": greedy_response.choices[0].message.content,
                    "usage": {
                        "completion_tokens": greedy_response.usage.completion_tokens,
                        "prompt_tokens": greedy_response.usage.prompt_tokens,
                    },
                }
            elif args.stop_at_n_unique_valid_samples == -1:
                # No early-stopping, let's get all samples at a time
                assert args.max_samples > 1
                if args.mock:
                    return {
                        "response": "",
                        "usage": {
                            "prompt_tokens": num_tokens_from_messages(
                                message, "gpt-4o-2024-05-13"
                            )
                            if count == 1
                            else 0,
                        },
                    }
                if sample_responses is not None:
                    # Directly return earlier samples
                    return {
                        "response": sample_responses.choices[count - 1].message.content,
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                if not non_sample_n:
                    assert count == 1
                    config = create_chatgpt_config(
                        message=message,
                        max_tokens=1024,
                        temperature=0.8,
                        batch_size=args.max_samples - 1,  # minus the 1 greedy sample
                        model=args.model,  # use gpt-4o for now.
                    )

                    sample_responses = request_chatgpt_engine(config)
                    return {
                        # "response": sample_responses.choices[0].message.content,
                        "response": sample_responses.choices[count - 1].message.content,
                        "usage": {
                            "completion_tokens": sample_responses.usage.completion_tokens,
                            "prompt_tokens": sample_responses.usage.prompt_tokens,
                        },
                    }
                else:
                    config = create_chatgpt_config(
                        message=message,
                        max_tokens=1024,
                        temperature=0.8,
                        # batch_size=args.max_samples - 1,  # minus the 1 greedy sample
                        model=args.model,  # use gpt-4o for now.
                    )

                    response = request_chatgpt_engine(config)
                    return {
                        "response": response.choices[0].message.content,
                        # "response": sample_responses.choices[count - 1].message.content,
                        "usage": {
                            "completion_tokens": response.usage.completion_tokens,
                            "prompt_tokens": response.usage.prompt_tokens,
                        },
                    }

        count = 0
        while count < args.max_samples:
            print(f"trying the {count + 1}-th sample ...")
            if 'claude' in args.model:
                ret = get_response(count, True)
            else:
                ret = get_response(count)
            count += 1
            traj.append(
                {
                    **ret,
                    "prompt": message,
                }
            )

            if args.mock:
                continue
            raw_output = ret["response"]
            logger.info(f"raw output:\n{raw_output}")
            all_generations.append(raw_output)

            # Let's only edit the first file in the edit commands.
            edited_file, new_content = _post_process_multifile_repair(
                raw_output,
                file_contents,
                file_loc_intervals,
                diff_format=args.diff_format,
            )

            if new_content == "":
                prev_contents.append("")
                file_names.append("")
            else:
                prev_content = file_contents[edited_file]
                prev_contents.append(prev_content)
                file_names.append(edited_file)

        counts.append(count)
        raw_outputs.append(raw_output)
        all_generations = [all_generations]
        prev_contents = [prev_contents]
        file_names = [file_names]

        res_data = {
            "instance_id": instance_id,
            "raw_output": raw_outputs,
            "all_generations": all_generations,
            "try_count": counts,
            "traj": traj,
            "prev_content": prev_contents,
            "file_names": file_names,
        }
        with output_file_lock:
            save_data_to_file(res_data, args.output_file)
        # break
    

def repair(args):
    logging.basicConfig(
        # filename=f"{args.output_folder}/repair.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.output_folder}/repair.log"),
            logging.StreamHandler()
        ]
    )
        
    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    if args.eval_n_limit:
        swe_bench_data = swe_bench_data.select(range(0, args.eval_n_limit))
        # locs = locs[:args.eval_n_limit]
    selected_ids = [data["instance_id"] for data in swe_bench_data]
    
    locs = load_jsonl(args.loc_file)
        
    if os.path.exists(args.output_file):
        prev_o = load_jsonl(args.output_file)
        processed_intances = [o["instance_id"] for o in prev_o]
    else:
        processed_intances = []
        
    # make copy of loc in output_folder
    with open(f"{args.output_folder}/used_locs.jsonl", "w") as f:
        for loc in locs:
            f.write(json.dumps(loc) + "\n")

    manager = mp.Manager()
    queue = manager.Queue()
    output_file_lock = manager.Lock()
    num_locs = 0
    
    for loc in locs:
        instance_id = loc["instance_id"]
        if instance_id in FILTER_INSTANCES: continue
        # if instance_id != 'astropy__astropy-12907':
        #     continue
        if instance_id not in processed_intances and instance_id in selected_ids:
            queue.put(loc)
            num_locs += 1
    
    log_queue = manager.Queue()
    queue_listener = QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()
    mp.spawn(
        run_repair_issue,
        nprocs=min(num_locs, args.num_processes) if args.num_processes > 0 else num_locs,
        args=(args, queue, log_queue, output_file_lock, swe_bench_data),
        join=True
    )
    queue_listener.stop()
            
        # found = False
        # for o in prev_o:
        #     if o["instance_id"] == instance_id:
        #         found = True
        #         break

        # if found:
        #     logging.info(f"skipping {instance_id} since patch already generated")
        #     continue

        # logging.info(f"================ repairing {instance_id} ================")

        


def post_process_raw_output(raw_output_text, file_contents, file_loc_intervals, args):
    git_diffs = ""
    raw_git_diffs = ""
    lint_success = False
    content = ""
    try:
        edited_file, new_content = _post_process_multifile_repair(
            raw_output_text,
            file_contents,
            file_loc_intervals,
            diff_format=args.diff_format,
        )
        if edited_file in file_contents:
            content = file_contents[edited_file]

            git_diff = fake_git_repo("playground", edited_file, content, new_content)

            raw_git_diffs += "\n" + git_diff.replace(
                r"\ No newline at end of file\n", ""
            )

            syntax_success = check_syntax(new_content)
            lint_success, prev_errors, errors = lint_code(
                "playground", "test.py", new_content, file_contents[edited_file]
            )

            differ_by_empty_lines = check_code_differ_by_just_empty_lines(
                new_content, file_contents[edited_file]
            )

            print(lint_success, prev_errors, errors, differ_by_empty_lines)

            if syntax_success and not differ_by_empty_lines:
                git_diffs = raw_git_diffs
            else:
                git_diffs = ""  # no need to evaluate
        else:
            diff = list(
                unified_diff(
                    content.split("\n"),
                    new_content.split("\n"),
                    fromfile=edited_file,
                    tofile=edited_file,
                    lineterm="",
                )
            )
            print("Failed parsing diff!")
            print("\n".join(diff))
    except Exception as e:
        print(raw_output_text)
        print(e)

    return git_diffs, raw_git_diffs, content


def post_process_repair(args):
    """
    apply some diff formatting.
    """
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    if args.process_n_limit > 0:
        raw_outputs = load_jsonl(args.raw_output_file)[:args.process_n_limit]
        locs = load_jsonl(args.loc_file)[:args.process_n_limit]
    else:
        raw_outputs = load_jsonl(args.raw_output_file)
        locs = load_jsonl(args.loc_file)

    for raw_output in raw_outputs:
        instance_id = raw_output["instance_id"]

        if raw_output["raw_output"] == "":
            with open(args.output_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "model_name_or_path": "auto_search_agent",
                            "instance_id": instance_id,
                            "model_patch": "",
                        }
                    )
                    + "\n"
                )
            continue

        if args.select_id == -1:
            # Use the last generation
            assert False, "not implemented for now"
        else:
            # Use the indexed generation
            generation_idx = args.select_id
            try:
                raw_output_text = raw_output["all_generations"][0][generation_idx]
                original_file_content = raw_output["prev_content"][0][generation_idx]
                pred_file = raw_output["file_names"][0][generation_idx]

                pred_files = [loc for loc in locs if loc["instance_id"] == instance_id][
                    0
                ]["found_files"][: args.top_n]

                git_diffs = ""
                raw_git_diffs = ""
                if isinstance(raw_output["raw_output"], str):
                    # for backward compatibility
                    raw_output["raw_output"] = [raw_output["raw_output"]]

                file_contents = {pred_file: original_file_content}

                file_loc_intervals = dict()

                loc = [loc for loc in locs if loc["instance_id"] == instance_id][0]
                if args.loc_agent_output_format:
                    G = pickle.load(
                        open(f"{DEPENDENCY_GRAPH_LOC}/{instance_id}.pkl", "rb")
                    )
                    searcher = RepoEntitySearcher(G)
                    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
                    problem_statement = bench_data["problem_statement"]
                    structure = get_repo_structure(
                        instance_id, bench_data["repo"], bench_data["base_commit"], "playground"
                    )

                    files, classes, _ = get_full_file_paths_and_classes_and_functions(structure)
        
                    pred_files, found_entities = [], []
                    for entity in loc['found_entities']:      
                        if not searcher.has_node(entity): continue
                        entity_file = entity.split(':')[0].strip()
                        if entity_file not in pred_files:
                            pred_files.append(entity_file)
                        if entity not in found_entities:
                            found_entities.append(entity)
                    
                    # pred_files = []
                    import collections
                    file_to_edit_locs = collections.defaultdict(list)
                    found_entities = found_entities[: args.top_n]
                    entity_datas = searcher.get_node_data(found_entities)
                    
                    for entity in entity_datas:
                        entity_file = entity['node_id'].split(':')[0].strip()
                        entity_name = entity['node_id'].split(':')[-1].strip()
                        if entity['type'] == 'function':
                            file_to_edit_locs[entity_file].append(f"function: {entity_name}")
                        else:
                            cls_data = [
                                cls_ for cls_ in classes 
                                if cls_['name']==entity_name and cls_['file']==entity_file
                            ]
                            if not cls_data: continue
                            if "methods" not in cls_data[0]:
                                file_to_edit_locs[entity_file].append(f"class: {entity_name}")
                                
                            init_func = [
                                method
                                for method in cls_data[0]["methods"]
                                if method["name"] == "__init__"
                            ]
                            if init_func:
                                file_to_edit_locs[entity_file].append(f"function: {entity_name}.__init__")
                            else:
                                file_to_edit_locs[entity_file].append(f"class: {entity_name}")
                                
                    if pred_file in file_to_edit_locs:
                        line_locs, context_intervals = transfer_arb_locs_to_locs(
                                file_to_edit_locs[pred_file],
                                None,
                                pred_file,
                                args.context_window,
                                args.loc_interval,
                                args.fine_grain_loc_only,
                                file_content=file_contents[pred_file]
                                if pred_file in file_contents
                                else "",
                            )
                    else:
                        line_locs, context_intervals = [], []  # default values.
                        
                    file_loc_intervals[pred_file] = context_intervals
                else:
            
                    for i, tmp_pred_file in enumerate(pred_files):
                        if tmp_pred_file != pred_file:
                            continue
                        if "found_edit_locs" in loc and len(loc["found_edit_locs"]) > i:
                            line_locs, context_intervals = transfer_arb_locs_to_locs(
                                loc["found_edit_locs"][i],
                                None,
                                loc["found_files"][i],
                                args.context_window,
                                args.loc_interval,
                                args.fine_grain_loc_only,
                                file_content=file_contents[pred_file]
                                if pred_file in file_contents
                                else "",
                            )
                        else:
                            line_locs, context_intervals = [], []  # default values.

                        file_loc_intervals[pred_file] = context_intervals
            except:
                raw_output_text = ""

        if raw_output_text:
            git_diffs, raw_git_diffs, content = post_process_raw_output(
                raw_output_text, file_contents, file_loc_intervals, args
            )
        else:
            git_diffs = ""
            raw_git_diffs = ""
            content = ""

        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": "auto_search_agent",
                        "instance_id": instance_id,
                        "model_patch": git_diffs.lstrip(),
                        "raw_model_patch": raw_git_diffs.lstrip(),
                        "original_file_content": content,
                    }
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_file", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--loc_interval", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument(
        "--stop_at_n_unique_valid_samples",
        type=int,
        default=-1,
        help="Early stop when we get N unique valid samples, set to -1 if don't want to do early stopping.",
    )
    parser.add_argument("--gen_and_process", action="store_true")
    parser.add_argument("--max_samples", type=int, default=20, help="Sampling budget.")
    parser.add_argument(
        "--select_id",
        type=int,
        default=-1,
        help="Index the selected samples during post-processing.",
    )
    parser.add_argument(
        # "--model", type=str, default="gpt-4o-2024-05-13", choices=["gpt-4o-2024-05-13"]
        "--model", type=str, default="gpt-4o", 
        choices=["openai/gpt-4o-2024-05-13", 
                "azure/gpt-4","azure/gpt-4o","azure/gpt-35-turbo", 
                "gpt-4o","gpt-35-turbo",
                "litellm_proxy/gpt-4o-2024-05-13",
                "litellm_proxy/claude-3-5-sonnet-20241022",
        ]
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--only_correct", action="store_true"
    )  # only work on correct loc files (saves time)
    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--fine_grain_loc_only", action="store_true")
    parser.add_argument("--diff_format", action="store_true")
    parser.add_argument("--skip_greedy", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument("--use_solution", action="store_true")
    parser.add_argument("--use_analysis", action="store_true")
    parser.add_argument("--summary_file", type=str, default="")
    parser.add_argument("--process_n_limit", type=int, default=-1)
    parser.add_argument("--repo_graph", action="store_true")

    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--log_level", type=str, default='INFO')
    parser.add_argument("--loc_agent_output_format", action="store_true")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.output_file = os.path.join(args.output_folder, "output.jsonl")

    if args.post_process:
        args.raw_output_file = args.output_file
        if args.select_id == -1:
            for i in range(args.max_samples):
                args.output_file = args.raw_output_file.replace(
                    ".jsonl", f"_{i}_processed.jsonl"
                )
                args.select_id = i
                post_process_repair(args)
        else:
            args.output_file = args.raw_output_file.replace(
                ".jsonl", f"_{args.select_id}_processed.jsonl"
            )
            post_process_repair(args)
    elif args.gen_and_process:
        repair(args)
        args.raw_output_file = args.output_file
        for i in range(args.max_samples):
            args.output_file = args.raw_output_file.replace(
                ".jsonl", f"_{i}_processed.jsonl"
            )
            args.select_id = i
            post_process_repair(args)
    else:
        repair(args)


if __name__ == "__main__":
    main()
