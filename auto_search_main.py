import argparse
import os
import json
import pickle
import logging
import logging.handlers
import time
# from functools import wraps
import toml
from queue import Empty
from typing import List
# import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from copy import deepcopy

from utils.util import convert_to_json
from utils.process_output import (
    # get_loc_edit_dict_from_raw_output,
    parse_raw_loc_output,
    get_loc_results_from_raw_outputs,
    get_loc_results_from_raw_outputs_v2,
    convert_to_loc_edit_list,
    merge_sample_locations,
    parse_keyword_json_obj,
    get_edit_modules_from_file_to_dict, 
    get_module_from_line_number
)

from utils.runtime.execute_ipython import execute_ipython
from utils.runtime import function_calling

from utils.cost_analysis import calc_cost
from plugins import LocationToolsRequirement
from plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
    get_current_issue_data,
    get_current_repo_modules
)
from utils.actions.action_parser import ResponseParser
from utils.actions.action import Action, ActionType

from utils.prompts.prompt import PromptManager
from utils.prompts import general_prompt
from utils.prompts.pipelines import (
    simple_localize_pipeline as simple_loc,
    reconstruct_then_localize_pipeline as recon_then_loc,
    module_then_bug_loc_pipeline as module_then_bug_loc,
    auto_search_prompt as auto_search,
    postprocess_loc_result,
)

from agentless.util.utils import load_jsonl
from agentless.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    transfer_arb_locs_to_locs
)
from evaluation.cal_module_recall import test_file_to_edit_locs
import litellm
from litellm import Message as LiteLLMMessage
from openai import APITimeoutError
from openai import OpenAI
import openai
client = OpenAI(
    base_url=os.environ['OPENAI_API_BASE'],
    api_key=os.environ['OPENAI_API_KEY']
)

from time import sleep
from concurrent.futures import ThreadPoolExecutor, TimeoutError
# from multiprocessing import Process, Queue, Lock, Pool, Manager
# from pathos.multiprocessing import ProcessPool
# import multiprocessing as mp
import torch.multiprocessing as mp
from utils.runtime.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
    STOP_WORDS as NON_FNCALL_STOP_WORDS
)
# litellm.set_verbose=True
# os.environ['LITELLM_LOG'] = 'DEBUG


def filter_dataset(dataset, filter_column: str, used_list: str):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            if used_list in data:
                selected_ids = data[used_list]
                logging.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                def filter_function(example):
                    return example[filter_column] in selected_ids  # Replace 'id' with the actual field name in the dataset
                filtered_dataset = dataset.filter(filter_function)
                # subset = dataset[dataset[filter_column].isin(selected_ids)]
                logging.info(f'Retained {len(filtered_dataset)} tasks after filtering')
                return filtered_dataset
    return dataset

# 186         - Call `search_interactions_among_modules` to analyze the interactions between all found files, classes or functions. And select the relevant paths to do further analysis.
# (system_prompt) Call `search_interactions_among_modules` to analyze the interactions between specified modules(files, classes or functions).


def get_all_valid_files():
    files, _, _ = get_current_repo_modules()
    all_valid_files = []
    for file_content in files:
        file = file_content[0]
        all_valid_files.append(file)
    return all_valid_files


def get_task_instruction(instance: dict, task: str = 'auto_search', include_pr=False, include_hint=False):
    output_format = None
    # for auto-search pipeline
    if task.strip() == 'auto_search':
        task_description = auto_search.TASK_INSTRUECTION_V0_1.format(
            package_name=instance['instance_id'].split('_')[0]
        )
        # task_description = auto_search.TASK_INSTRUECTION_V1_1
    
    # for module_loc_then_bug_loc pipeline
    elif task.strip() == 'pr_rewrite':
        task_description = module_then_bug_loc.REWRITE_PR_TASK
    
    elif task.strip() == 'keyword_extraction':
        # KEYWORD_EXTRACTION_TASK.format(package_name=instance_id.split('_')[0]) + \
                                # OUTPUT_FORMAT_KEYWORD_EXTRACTION
        task_description = module_then_bug_loc.KEYWORD_EXTRACTION_TASK.format(package_name=instance['instance_id'].split('_')[0])
        output_format = module_then_bug_loc.OUTPUT_FORMAT_KEYWORD_EXTRACTION
        
    elif task.strip() == 'modules_localization':
        task_description = module_then_bug_loc.FULL_QUALIFIED_NAME_INFER_AFTER_EXTRACTION_TASK
        
    elif task.strip() == 'bug_localization':
        task_description = module_then_bug_loc.SEARCH_INSTRUCTION
        output_format = module_then_bug_loc.OUTPUT_FORMAT_LOC
    
    # elif task.strip() == 'keyword_extraction_then_modules_localization':
    #     # merge extraction with localize module
    #     task_description = module_then_bug_loc.FULL_QUALIFIED_NAME_INFER_TASK.format(
    #         package_name=instance['instance_id'].split('_')[0],
    #         output_format=module_then_bug_loc.OUTPUT_FORMAT_KEYWORD_EXTRACTION
    #     )
    
        
    # recon_then_loc pipeline
    elif task.strip() == 'reconstruct_then_localize':
        task_description = recon_then_loc.RECONSTRUCT_FLOW_TASK
        # task_description = RECONSTRUCT_FLOW_TASK_UNIFY
        # task_description = RECONSTRUCT_FLOW_TASK_GENERAL
        output_format = recon_then_loc.RECONSTRUCT_TAK_OUTPUT_FORMAT
        
    
    elif task.strip() == 'simple_localize':
        task_description = simple_loc.SEARCH_LOC_TASK_INSTRUCTION
        output_format = simple_loc.OUTPUT_FORMAT_LOC
        
    else:
        return None

    instruction = ""
    instruction += task_description
        
    if include_pr:
        problem_statement = instance['problem_statement']
        instruction += general_prompt.PR_TEMPLATE.format(
            title=problem_statement.strip().split('\n')[0],
            description = '\n'.join(problem_statement.strip().split('\n')[1:]).strip()
        )
    
    if output_format:
        instruction += output_format
    
    if include_hint:
        instruction += (
            'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
            'Don\'t include any lambda functions!\n'
            'You should NOT modify any files!\n'
        )

    # NOTE: You can actually set slightly different instruction for different task
    # instruction += AGENT_CLS_TO_INST_SUFFIX
    return instruction


def auto_search_process(result_queue,
                        model_name, messages, fake_user_msg,
                        tools = None,
                        traj_data=None,
                        temp=1,
                        max_iteration_num=20):
    
    if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower() 
                #   or model_name=='azure/gpt-4o' 
                #   or model_name == 'litellm_proxy/o3-mini-2025-01-31'
                  ):
        # 转换message
        messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
            
    code_history = []
    parser = ResponseParser()
    if not traj_data:
        traj_msgs = messages.copy()
        prompt_tokens = 0
        completion_tokens = 0
    else:
        traj_msgs = traj_data['messages']
        prompt_tokens = traj_data['usage']['prompt_tokens']
        completion_tokens = traj_data['usage']['completion_tokens']
        
    cur_interation_num = 0
    last_message = None
    finish = False
    while not finish:
        cur_interation_num += 1
        if cur_interation_num == max_iteration_num:
            messages.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })
            traj_msgs.append({
                'role': 'user',
                'content': 'The Maximum number of interation has been reached, please generate your final output with required format and use <finish></finish> to exit.'
            })

        try:
            # new conversation
            if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower()):
                messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
                response = litellm.completion(
                    model=model_name,
                    # temperature=temp, top_p=0.8, repetition_penalty=1.05, 
                    temperature=0.7, top_p=0.8, repetition_penalty=1.05, 
                    messages=messages,
                    # temperature=temp,
                    stop=NON_FNCALL_STOP_WORDS
                )
            elif 'deepseek' in model_name:
                messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False, 
                    max_tokens=8192
                )
            # elif tools and model_name == 'azure/gpt-4o':
            #     messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
            #     response = litellm.completion(
            #         model=model_name,
            #         # tools=tools,
            #         messages=messages,
            #         temperature=temp,
            #         stop=NON_FNCALL_STOP_WORDS
            #     )
            elif tools and model_name == 'litellm_proxy/o3-mini-2025-01-31':
                # messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
                response = litellm.completion(
                    model=model_name,
                    tools=tools,
                    messages=messages,
                    temperature=temp,
                    # stop=NON_FNCALL_STOP_WORDS,
                    reasoning_effort= "high"
                )
            elif tools:
                # logging.info('===========================')
                # logging.info('using tools')
                # logging.info('===========================')
                response = litellm.completion(
                    model=model_name,
                    tools=tools,
                    messages=messages,
                    temperature=temp,
                    # stop=['</execute_ipython>'], #</finish>',
                )
            else:
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    stop=['</execute_ipython>'], #</finish>',
                )
                # response = client.chat.completions.create(
                #     model="ft:gpt-4o-mini-2024-07-18:personal::AH52kz8c",
                #     messages=messages,
                #     temperature=temp
                # )
        except litellm.BadRequestError as e:
            # If there's an error, send the error info back to the parent process
            result_queue.put({'error': str(e), 'type': 'BadRequestError'})
            return
        
        if last_message and response.choices[0].message.content == last_message:
            messages.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            traj_msgs.append({
                "role": "user",
                "content": "OBSERVATION:\n" + "Don't repeat your response.\n" + fake_user_msg,
            })
            continue
        
        raw_response = deepcopy(response)
        # logging.info('response.choices[0].message')
        # logging.info(response.choices[0].message)
        if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower()
                    #   or model_name=='azure/gpt-4o' 
                    #   or model_name == 'litellm_proxy/o3-mini-2025-01-31'
                      or 'deepseek' in model_name
                      ):
            try:
                non_fncall_response_message = response.choices[0].message
                fn_call_messages_with_response = (
                    convert_non_fncall_messages_to_fncall_messages(
                        [non_fncall_response_message], tools # messages + 
                    )
                )
                fn_call_response_message = fn_call_messages_with_response[-1]
                if not isinstance(fn_call_response_message, LiteLLMMessage):
                    fn_call_response_message = LiteLLMMessage(
                        **fn_call_response_message
                    )
                response.choices[0].message = fn_call_response_message
            except:
                logging.info('convert none fncall messages failed.')
                continue 
                
        last_message = response.choices[0].message.content
        print(response.choices[0].message)
        messages.append(convert_to_json(raw_response.choices[0].message))
        traj_msgs.append(convert_to_json(raw_response.choices[0].message))
        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens  
            
        actions = parser.parse(response)
        if not isinstance(actions, List):
            actions = [actions]
        for action in actions:
            logging.debug(action.action_type)
            if action.action_type == ActionType.FINISH:
                final_output = action.thought
                # traj_msgs.append({
                #     'role': 'assistant',
                #     'content': response.choices[0].message.content,
                #     'action_type': action.action_type,
                # })
                logging.info('='*15)
                logging.info("\nFinal Response:=\n" + final_output)
                finish = True # break
            elif action.action_type == ActionType.MESSAGE:
                logging.debug("thought:\n" + action.content)
                # traj_msgs.append({
                #     'role': 'assistant',
                #     'content': response.choices[0].message.content,
                #     'action_type': action.action_type,
                # })
                # check if enough
                messages.append({"role": "user", "content": fake_user_msg})
                traj_msgs.append({"role": "user", "content": fake_user_msg})
                # continue
            elif action.action_type == ActionType.RUN_IPYTHON:
                ipython_code = action.code.strip('`')
                logging.debug('code:\n' + ipython_code)
                # if not ipython_code:
                #     traj_msgs.append({
                #         'role': 'assistant',
                #         'content': response.choices[0].message.content,
                #         'action_type': action.action_type,
                #         'code': ipython_code,
                #     })
                #     messages.append({
                #         "role": "user",
                #         "content": "OBSERVATION:\n" + "Warn: Empty code.",
                #     })
                #     traj_msgs.append({
                #         "role": "user",
                #         "content": "OBSERVATION:\n" + "Warn: Empty code.",
                #     })
                #     continue
                # elif ipython_code in code_history:
                #     traj_msgs.append({
                #         'role': 'assistant',
                #         'content': response.choices[0].message.content,
                #         'action_type': action.action_type,
                #         'code': ipython_code,
                #     })
                #     messages.append({
                #         "role": "user",
                #         "content": "OBSERVATION:\n" + "The query has already existed in the history dialog, please give a new query or send your final answer.",
                #     })
                #     traj_msgs.append({
                #         "role": "user",
                #         "content": "OBSERVATION:\n" + "The query has already existed in the history dialog, please give a new query or send your final answer.",

                #     })
                #     continue
                # else:
                #     code_history.append(ipython_code)

                # for ipython_code in [ipython_code]:

                # if ipython_code:
                logging.info(f"Executing code:\n```\n{ipython_code}\n```")
                # print(f"Executing code:\n```\n{ipython_code}\n```")
                function_response = execute_ipython(ipython_code)
                try:
                    function_response = eval(function_response)
                except SyntaxError:
                    function_response = function_response
                if not isinstance(function_response, str):
                    function_response = str(function_response)
                    
                # print("OBSERVATION:\n" + function_response)
                logging.info("OBSERVATION:\n" + function_response)
                if not tools:
                    messages.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "user",
                        "content": "OBSERVATION:\n" + function_response,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
                    traj_msgs.append({
                        "role": "tool",
                        "tool_call_id": action.tool_call_id,
                        "name": action.function_name,
                        "content": "OBSERVATION:\n" + function_response,
                    })
            else:
                logging.warning('Error Action!')
                # return

    # save traj
    traj_data = {
        'messages': traj_msgs,
        'tools': tools,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
    }
    # return final_output, messages, traj_data
    result_queue.put((final_output, messages, traj_data))


def save_data_to_file(data: dict, file: str):
    with open(file, "a") as f:
        f.write(
            json.dumps(data) + "\n"
        )


def run_localize(rank, args, bug_queue, log_queue, output_file_lock, traj_file_lock):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")

    # res = {'message': f'------ rank {rank} start ------'}
    # with traj_file_lock:
    #     save_data_to_file(res, os.path.join(args.output_folder, 'loc_trajs.jsonl'))

    # try:
    #     bug = bug_queue.get_nowait()
    #     # print(rank, bug['instance_id'])
    #     set_current_issue(instance_data=bug, rank=rank)
    # except Empty:
    #     print(rank, 'finished')


    while True:
        try:
            bug = bug_queue.get_nowait()
        except Empty:
            break

        instance_id = bug["instance_id"]
        prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'utils/prompts'),
            agent_skills_docs=LocationToolsRequirement.documentation,
        )

        logger.info("=" * 60)
        logger.info(f"==== rank {rank} setup localize {instance_id} ====")
        set_current_issue(instance_data=bug)
        
        issue_id, bench_data, structure = get_current_issue_data()
        problem_statement = bug["problem_statement"]
        all_valid_files = get_all_valid_files()

        # loc result template
        found_files = []
        found_edit_locs = []
        raw_output_loc = []
        additional_artifact_loc_edit_location = None

        loc_trajs = {'trajs': []}
        total_prompt_tokens, total_completion_tokens = 0, 0

        for _ in range(args.num_samples):
            logger.info("=" * 60)
            logger.info(f"==== rank {rank} begin localizing {instance_id} ====")
            max_attempt_num = args.max_attempt_num
            while max_attempt_num:
                logger.info("=" * 60)
                logger.info(f"==== {instance_id} Count down: attempt {max_attempt_num} ====")
                loc_start_time = time.time()
                try:
                    """
                    Basic instructions:
                        - CodeAct instruction
                        - Few-shot Examples
                    """
                    if args.use_function_calling:
                        system_prompt = function_calling.SYSTEM_PROMPT
                        # system_prompt = CLAUDE_THINKING_INSTRUCTION
                    else:
                        system_prompt = prompt_manager.system_message
                        
                    messages: list[dict] = [{
                        "role": "system",
                        "content": system_prompt
                    }]
                    
                    if args.selected_pipeline == 'module_loc_then_bug_loc':
                        messages.append({
                            "role": "system",
                            "content": general_prompt.SYSTEM_PROMPT
                        })
                    # elif args.selected_pipeline == 'auto_search':
                    #     messages.append({
                    #         "role": "system",
                    #         "content": auto_search.SYSTEM_PROMPT
                    #     })
                        
                    if args.use_example:
                        messages.append({
                            "role": "user",
                            "content": prompt_manager.initial_user_message
                        })

                    if args.selected_pipeline == 'auto_search':
                        logger.info(f"==== {instance_id} start auto search ====")
                        messages.append({
                            "role": "user",
                            "content": get_task_instruction(bug, include_pr=True, include_hint=True),
                        })
                        
                        ctx = mp.get_context('fork')  # use fork to inherit context!!
                        result_queue = ctx.Manager().Queue()
                        tools = None
                        if args.use_function_calling:
                            tools = function_calling.get_tools(
                                codeact_enable_search_keyword=True,
                                codeact_enable_search_entity=True,
                                # codeact_enable_fragmental_content_tools=True,
                                
                                codeact_enable_tree_structure_traverser=True,
                                # codeact_enable_graph_structure_traverser=True,
                                # codeact_enable_specific_hops_structure_traverser=True,
                                # codeact_enable_fragmental_structure_tools=True,
                                simple_desc = args.simple_desc,
                            )
                        process = ctx.Process(target=auto_search_process, kwargs={
                            'result_queue': result_queue,
                            'model_name': args.model,
                            'messages': messages,
                            'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC_,
                            'temp': 1,
                            'tools': tools,
                        })
                        process.start()
                        process.join(timeout=args.timeout)
                        if process.is_alive():
                            logger.warning(f"{instance_id} attempt {max_attempt_num} execution flow "
                                           f"reconstruction exceeded timeout. Terminating.")
                            process.terminate()
                            process.join()
                            raise TimeoutError
                        
                        # loc_result, messages, traj_data = result_queue.get()
                        result = result_queue.get()
                        if isinstance(result, dict) and 'error' in result and result['type'] == 'BadRequestError':
                            raise litellm.BadRequestError(result['error'], args.model, args.model.split('/')[0])
                            # print(f"Error occurred in subprocess: {result['error']}")
                        else:
                            loc_result, messages, traj_data = result
                        
                    elif args.selected_pipeline == 'reconstruct_then_localize':
                        ## task1: reconstruct the execution flow
                        logger.info(f"==== {instance_id} start execution flow reconstruction ====")
                        messages.append({
                            "role": "user",
                            "content": get_task_instruction(bug,
                                                            task='reconstruct_then_localize',
                                                            include_pr=True,
                                                            include_hint=True,
                                                            ),
                        })

                        ctx = mp.get_context('fork')  # use fork to inherit context!!
                        result_queue = ctx.Manager().Queue()
                        process = ctx.Process(target=auto_search_process, kwargs={
                            'result_queue': result_queue,
                            'model_name': args.model,
                            'messages': messages,
                            'fake_user_msg': recon_then_loc.FAKE_USER_MSG_IN_RECONSTRUCT_TAK,
                            'temp': 1,
                        })
                        process.start()
                        process.join(timeout=args.timeout)
                        if process.is_alive():
                            logger.warning(f"{instance_id} attempt {max_attempt_num} execution flow "
                                           f"reconstruction exceeded timeout. Terminating.")
                            process.terminate()
                            process.join()
                            raise TimeoutError
                        
                        exec_flow_result, messages, cons_traj_data = result_queue.get()
                        # TODO: eval whether `exec_flow_result` is valid
                        # TODO: hint

                        ## task2: search for locations
                        logger.info(f"==== {instance_id} start localization ====")
                        loc_messages = messages[:-1].copy()
                        # remove finish sign to continue conversation
                        loc_messages.append({
                            "role": messages[-1]['role'],
                            "content": messages[-1]['content'][:-len("<finish></finish>")].rstrip()
                        })
                        loc_messages.append({"role": "user", "content": recon_then_loc.SEARCH_AFTER_RECONSTRUCTION_TASK})
                        cons_traj_data['messages'].append({"role": "user", "content": recon_then_loc.SEARCH_AFTER_RECONSTRUCTION_TASK})

                        result_queue = ctx.Manager().Queue()
                        process = ctx.Process(target=auto_search_process, kwargs={
                            'result_queue': result_queue,
                            'model_name': args.model,
                            'messages': loc_messages,
                            'traj_data': cons_traj_data,
                            'fake_user_msg': recon_then_loc.FAKE_USER_MSG_FOR_LOC,
                            'temp': 1,
                            'max_iteration_num': 10,
                        })
                        process.start()
                        process.join(timeout=args.timeout)
                        if process.is_alive():
                            logger.warning(f"{instance_id} attempt {max_attempt_num} localization "
                                           f"exceeded timeout. Terminating.")
                            process.terminate()
                            process.join()
                            raise TimeoutError
                        loc_result, messages, traj_data = result_queue.get()
                        
                    elif args.selected_pipeline == 'module_loc_then_bug_loc':
                        # logger.info(f"==== {instance_id} start execution flow reconstruction ====")
                        
                        # task1: PR rewrite
                        messages.append({"role": "user", "content": get_task_instruction(bug, task='pr_rewrite', include_pr=True)})
                        response = litellm.completion(
                            model=args.model,
                            messages=messages,
                            temperature=1,
                        )
                        messages.append(convert_to_json(response.choices[0].message))
                        logger.info(messages)
                        
                        # task1.5: KEYWORD_EXTRACTION_TASK
                        messages.append({
                            "role": "user",
                            "content": get_task_instruction(bug, task='keyword_extraction'),
                        })
                        response = litellm.completion(
                            model=args.model,
                            messages=messages,
                            temperature=1,
                        )
                        messages.append(convert_to_json(response.choices[0].message))
                        logger.info(messages)
                        
                        # automatedly process the extracted keywords/lines
                        # if args.inforce_search_keywords:
                        #     parsed_search_terms = parse_keyword_json_obj(response.choices[0].message.content)
                        #     from plugins.location_tools.repo_ops.repo_ops import search_repo_by_json_obj
                        #     search_result = search_repo_by_json_obj(parsed_search_terms)
                        #     messages.append({
                        #         "role": "user",
                        #         "content": search_result
                        #     })
                        
                        # task2: module localization
                        messages.append({
                            "role": "user",
                            "content": get_task_instruction(bug, task='modules_localization')
                                # FULL_QUALIFIED_NAME_INFER_AFTER_EXTRACTION_TASK
                        })
                        ctx = mp.get_context('fork')  # use fork to inherit context!!
                        result_queue = ctx.Manager().Queue()
                        tools = None
                        if args.use_function_calling:
                            tools = function_calling.get_tools(
                                codeact_enable_tree_structure_traverser=True
                            )
                            logger.info("=========================")
                            logger.info(f'tools: len({len(tools)})')
                            logger.info("=========================")
                        process = ctx.Process(target=auto_search_process, kwargs={
                            'result_queue': result_queue,
                            'model_name': args.model,
                            'messages': messages,
                            'fake_user_msg': module_then_bug_loc.FAKE_USER_MSG_FOR_MLOC,
                            'temp': 1,
                            'tools': tools,
                        })
                        process.start()
                        process.join(timeout=args.timeout)
                        if process.is_alive():
                            logger.warning(f"{instance_id} attempt {max_attempt_num} execution flow "
                                           f"reconstruction exceeded timeout. Terminating.")
                            process.terminate()
                            process.join()
                            raise TimeoutError
                        
                        exec_flow_result, messages, cons_traj_data = result_queue.get()
                        # TODO: eval whether `exec_flow_result` is valid
                        # TODO: hint

                        ## task2: search for locations
                        logger.info(f"==== {instance_id} start localization ====")
                        loc_messages = messages[:-1].copy()
                        # remove finish sign to continue conversation
                        loc_messages.append({
                            "role": messages[-1]['role'],
                            "content": messages[-1]['content'][:-len("<finish></finish>")].rstrip()
                        })
                        loc_messages.append({
                            "role": "user", 
                            "content": get_task_instruction(bug, task="bug_localization") # , include_pr=True
                        })
                        cons_traj_data['messages'].append({
                            "role": "user", 
                            # "content": SEARCH_INSTRUCTION
                            "content": get_task_instruction(bug, task="bug_localization") # , include_pr=True
                        })
                        tools = None
                        if args.use_function_calling:
                            tools = function_calling.get_tools(
                                # codeact_enable_search_call_relation=True,
                                # codeact_enable_explore_structure=True,
                                codeact_enable_tree_structure_traverser=True
                            )
                            logger.info("=========================")
                            logger.info(f'tools: len({len(tools)})')
                            logger.info("=========================")
                        result_queue = ctx.Manager().Queue()
                        process = ctx.Process(target=auto_search_process, kwargs={
                            'result_queue': result_queue,
                            'model_name': args.model,
                            'messages': loc_messages,
                            'traj_data': cons_traj_data,
                            'fake_user_msg': module_then_bug_loc.FAKE_USER_MSG_FOR_LOC,
                            'temp': 1,
                            'max_iteration_num': 10,
                            'tools': tools,
                        })
                        process.start()
                        process.join(timeout=args.timeout)
                        if process.is_alive():
                            logger.warning(f"{instance_id} attempt {max_attempt_num} localization "
                                           f"exceeded timeout. Terminating.")
                            process.terminate()
                            process.join()
                            raise TimeoutError
                        loc_result, messages, traj_data = result_queue.get()

                    elif args.selected_pipeline == 'simple_localize':
                        messages.append({
                            "role": "user",
                            "content": get_task_instruction(bug, task=args.selected_pipeline, include_pr=True),
                        })
                        tools = None
                        if args.use_function_calling:
                            tools = function_calling.get_tools(
                                # codeact_enable_explore_structure=True,
                                codeact_enable_local_structure_traverser=True
                            )
                        ctx = mp.get_context('fork')  # use fork to inherit context!!
                        result_queue = ctx.Manager().Queue()
                        process = ctx.Process(target=auto_search_process, kwargs={
                            'result_queue': result_queue,
                            'model_name': args.model,
                            'messages': messages,
                            'fake_user_msg': simple_loc.FAKE_USER_MSG_FOR_LOC,
                            'temp': 1,
                            'tools': tools,
                        })
                        process.start()
                        process.join(timeout=args.timeout)
                        if process.is_alive():
                            logger.warning(f"{instance_id} attempt {max_attempt_num} localization "
                                           f"exceeded timeout. Terminating.")
                            process.terminate()
                            process.join()
                            raise TimeoutError
                        loc_result, messages, traj_data = result_queue.get()
                        # loc_result, messages, traj_data = auto_search_process(model_name=args.model,
                        #                                                       messages=messages,
                        #                                                       fake_user_msg=FAKE_USER_MSG_FOR_LOC,
                        #                                                       temp=1)
                
                except litellm.BadRequestError as e:
                    logger.warning(f'{e}. Try again.')
                    continue
                except APITimeoutError:
                    logger.warning(f"APITimeoutError. Try again.")
                    sleep(10)
                    continue
                except TimeoutError:
                    logger.warning(f"Processing time exceeded 15 minutes. Try again.")
                    max_attempt_num = max_attempt_num - 1
                    continue
                except litellm.exceptions.ContextWindowExceededError as e:
                    logger.warning(f'{e}. Try again.')
                    max_attempt_num = max_attempt_num - 1
                    continue
                # except Exception as e:
                #     logger.warning(f"Error processing instance {instance_id}: {e}. Skipping to the next.")
                #     max_attempt_num = max_attempt_num - 1
                #     break

                loc_end_time = time.time()
                if not loc_result:
                    continue # empty result
                
                max_attempt_num = max_attempt_num - 1
                if max_attempt_num != 0:
                    # 用于生成训练数据
                    # 多次尝试机会：不是最后一次尝试，则计算是否成功
                    file_list, loc_edit_dict = parse_raw_loc_output(loc_result, all_valid_files)
                    found_edit_locs = convert_to_loc_edit_list(loc_edit_dict, file_list)
                    file_list = file_list[:3] # top 3 file
                    found_edit_locs = found_edit_locs[:3]
                    
                    file_to_edit_locs = dict()
                    for i, temp_edit_locs in enumerate(found_edit_locs):
                        file_to_edit_locs[file_list[i]] = temp_edit_locs
                    result = test_file_to_edit_locs(bug, file_list, file_to_edit_locs, use_module=True) # TODO: top_n
                    if not result[-1]:
                        logger.info(f"==== Find wrong lines, continue to localize. ====")
                        continue

                total_prompt_tokens += traj_data['usage']['prompt_tokens']
                total_completion_tokens += traj_data['usage']['completion_tokens']
                traj_data['time'] = loc_end_time - loc_start_time
                loc_trajs['trajs'].append(traj_data)

                # generate correct output or finish last attempt
                raw_output_loc.append(loc_result)
                break

        if not raw_output_loc:
            # loc generalization failed
            logger.info(f"==== localizing {instance_id} failed, save empty outputs ====")
            loc_res = {
                "instance_id": instance_id,
                "found_files": [[]],
                "found_files": [[]],
                "found_edit_locs": [[]],
                # "found_locs": [[]],
                # "found_valid_locs" : [[]],
                "additional_artifact_loc_edit_location": []
            }
            with output_file_lock:
                save_data_to_file(loc_res, args.output_file)
        else:
            # process multiple loc outputs
            logger.info(f"==== localizing {instance_id} succeed, process multiple loc outputs ====")

            found_files, found_edit_locs = get_loc_results_from_raw_outputs(raw_output_loc, all_valid_files)
            # found_files, found_locs, found_valid_locs = get_loc_results_from_raw_outputs_v2(
            #     raw_output_loc, all_valid_files, structure)

            loc_res = {
                "instance_id": instance_id,
                "found_files": found_files,
                "found_edit_locs": found_edit_locs,
                "additional_artifact_loc_edit_location": [{'raw_output_loc': raw_output_loc}],
                "meta_data": {
                    'repo': bug['repo'],
                    'base_commit': bug['base_commit'],
                    'problem_statement': bug['problem_statement'],
                    'patch': bug['patch']
                }
            }
            with output_file_lock:
                save_data_to_file(loc_res, args.output_file)

            cost = calc_cost(args.model, total_prompt_tokens, total_completion_tokens)
            loc_res['usage'] = {'cost($)': f'{round(cost, 5)}', 'prompt_tokens': total_prompt_tokens,
                                'completion_tokens': total_completion_tokens}
            loc_res['loc_trajs'] = loc_trajs
            traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
            with traj_file_lock:
                save_data_to_file(loc_res, traj_file)

        reset_current_issue()

def clear_file(file_path):
    with open(file_path, 'w') as f:
        f.write("")

def backup_file(original_file):
    backup_path = original_file
    if os.path.exists(original_file):
        import shutil
        # Define the backup file path
        backup_path = original_file + '.backup'
        
        # Copy the original file to create a backup
        shutil.copy2(original_file, backup_path)
    return backup_path

def delete_file(file_path):
    try:
        # Delete the file
        os.remove(file_path)
        # print(f"File '{file_path}' has been deleted successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied: Cannot delete '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")
    
def localize(args):
    if args.loc_bench:
        selected_ids = []
        selected_ins_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
        if os.path.exists(selected_ins_file_path):
            with open(selected_ins_file_path, 'r') as sf:
                data = toml.load(sf)
                if args.used_list in data:
                    selected_ids = data[args.used_list]
                
        loc_bench_data = []
        with open(args.dataset, 'r') as dtf:
            for line in dtf:
                instance = json.loads(line)
                if not selected_ids:
                    loc_bench_data.append(instance)
                if selected_ids and instance['instance_id'] in selected_ids:
                    loc_bench_data.append(instance)
                    
        swe_bench_tests = loc_bench_data
        if args.eval_n_limit:
            eval_n_limit = min(args.eval_n_limit, len(swe_bench_tests))
            swe_bench_tests = swe_bench_tests[:eval_n_limit]
    else:
        swe_bench_data = load_dataset(args.dataset, split=args.split)
        swe_bench_tests = filter_dataset(swe_bench_data, 'instance_id', args.used_list)
        if args.eval_n_limit:
            eval_n_limit = min(args.eval_n_limit, len(swe_bench_tests))
            swe_bench_tests = swe_bench_tests.select(range(0, eval_n_limit))
            logging.info(f'Limiting evaluation to first {eval_n_limit} instances.')

    manager = mp.Manager()
    queue = manager.Queue()
    output_file_lock, traj_file_lock = manager.Lock(), manager.Lock()

    # collect processed instances
    processed_instance = []
    if os.path.exists(args.output_file):
        traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
        locs = load_jsonl(args.output_file)        
        if args.rerun_empty_location:
            traj_datas = load_jsonl(traj_file)
            backup_loc_output = backup_file(args.output_file)
            backup_traj_output = backup_file(traj_file)
            clear_file(args.output_file)
            clear_file(traj_file)
            for loc in locs:
                if loc['found_files'] != [[]]:
                    save_data_to_file(loc, args.output_file)
                    processed_instance.append(loc['instance_id'])
                    
            for loc_traj in traj_datas:
                if loc_traj['found_files'] != [[]]:
                    save_data_to_file(loc_traj, traj_file)
        else:
            processed_instance = [loc['instance_id'] for loc in locs]
    
    num_bugs = 0
    for bug in swe_bench_tests:
        instance_id = bug["instance_id"]
        if instance_id in processed_instance:
            print(f"instance {instance_id} has already been processed, skip.")
        else:
            queue.put(bug)
            num_bugs += 1

    log_queue = manager.Queue()
    queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()
    mp.spawn(
        run_localize,
        nprocs=min(num_bugs, args.num_processes) if args.num_processes > 0 else num_bugs,
        args=(args, queue, log_queue, output_file_lock, traj_file_lock),
        join=True
    )
    queue_listener.stop()
    
    if args.rerun_empty_location:
        try:
            delete_file(backup_loc_output)
            delete_file(backup_traj_output)
        except:
            return


def merge(args):
    args.merge_file = 'merged_' + args.output_file.split('/')[-1].split('.')[0] + f'_{args.ranking_method}.jsonl'
    # args.merge_file = 'merged_' + args.output_file.split('/')[-1]
    # args.merge_file = args.merge_file.replace('.jsonl', f'_{args.num_samples}.jsonl')
    args.merge_file = os.path.join(args.output_folder, args.merge_file)
    with open(args.merge_file, "w") as file:
        file.write("")

    with open(args.output_file, 'r') as file:
        for line in file:
            loc_data = json.loads(line)
            if loc_data['found_files'] == [[]]:
                loc_data['found_files'] = []
                loc_data['found_edit_locs'] = [[]]
            else:
                merged_fils, merged_edit_locs = merge_sample_locations(loc_data['found_files'], 
                                                                    loc_data['found_edit_locs'],
                                                                    ranking_method=args.ranking_method,
                                                                    )
                loc_data['found_files'] = merged_fils
                loc_data['found_edit_locs'] = merged_edit_locs
            with open(args.merge_file, 'a') as f:
                f.write(json.dumps(loc_data) + '\n')


def retrieve_graph(code_graph, graph_tags, search_term, structure):
    one_hop_modules = dict()
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
                if tag['rel_fname'] not in one_hop_modules:
                    one_hop_modules[tag['rel_fname']] = dict()
                module_name = f"function: {txt['name']}"
                if module_name not in one_hop_modules[tag['rel_fname']]:
                    one_hop_modules[tag['rel_fname']][module_name] = {
                        'start_line': txt['start_line'],
                        'end_line': txt['end_line']
                    }

        for txt in s['classes']:
            for func in txt['methods']:
                if tag['line'] >= func['start_line'] and tag['line'] <= func['end_line']:
                    if tag['rel_fname'] not in one_hop_modules:
                        one_hop_modules[tag['rel_fname']] = dict()
                    module_name = f"function: {txt['name']}.{func['name']}"
                    if module_name not in one_hop_modules[tag['rel_fname']]:
                        one_hop_modules[tag['rel_fname']][module_name] = {
                            'start_line': func['start_line'],
                            'end_line': func['end_line']
                        }

    return one_hop_modules


def construct_one_hop_dependencies(found_related_locs, code_graph, graph_tags, structure):
    one_hop_dependencies = [[] for _ in found_related_locs]
    # retrieve the code graph for dependent functions and classes
    for i, items in enumerate(found_related_locs):
        searched_locs = []
        for item in items:
            item = item.splitlines()
            for loc in item:
                init_loc = loc
                # class_loc = None
                if loc in searched_locs:
                    continue
                else:
                    searched_locs.append(loc)

                if loc.startswith("class: ") and "." not in loc:
                    loc = loc[len("class: ") :].strip()
                elif loc.startswith("function: ") and "." not in loc:
                    loc = loc[len("function: ") :].strip()
                elif loc.startswith("method: ") and "." not in loc:
                    loc = loc[len("method: ") :].strip()
                elif "." in loc:
                    loc = loc.split(".")[-1].strip()
                    # class_loc = loc.split(".")[0].split(':')[-1].strip()
                else:
                    continue

                modules = retrieve_graph(code_graph, graph_tags, loc, structure)
                if modules:
                    one_hop_dependencies[i].append({
                        init_loc: modules
                    })

                # if class_loc not in searched_locs:
                #     init_class_loc = f'class: {class_loc}'
                #     class_dependencies = retrieve_graph(code_graph, graph_tags, class_loc, structure)
                #     if class_dependencies:
                #         one_hop_dependencies[i].append({
                #             init_class_loc: modules
                #         })
    return one_hop_dependencies


def add_dependencies(args):
    locs = load_jsonl(args.output_file)
    if args.expand_context:
        output_file_name = args.output_file.replace(".jsonl", f"_exp_repo_graph.jsonl")
    else:
        output_file_name = args.output_file.replace(".jsonl", f"_repo_graph.jsonl")
    
    with open(output_file_name, "w") as file:
        file.write("")

    for loc in tqdm(locs):
        instance_id = loc["instance_id"]
        pred_files = loc["found_files"]
        if len(loc["found_files"]) == 0:
            print(f"{instance_id}'s found_files is emptry.")
            continue

        # grab structure data
        set_current_issue(instance_id)
        issue_id, bench_data, structure = get_current_issue_data()
        files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
        
        # Construct top-n file context
        file_to_edit_locs = dict()
        for i, pred_file in enumerate(pred_files):
            if "found_edit_locs" in loc and len(loc["found_edit_locs"]) > i:
                file_to_edit_locs[pred_file] = loc["found_edit_locs"][i]
        
        # add dependencies
        if args.expand_context:
            # add +/- 10 lines to the context

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

            # found_edit_locs -> file_loc_intervals
            file_loc_intervals = dict()
            for pred_file, locs in file_to_edit_locs.items():
                content = file_contents[pred_file]
                line_locs, context_intervals = transfer_arb_locs_to_locs(
                    locs,
                    structure,
                    pred_file,
                    loc_interval=True,
                    file_content=file_contents[pred_file] if pred_file in file_contents else "",
                )

                if len(line_locs) > 0:
                    lines = content.split("\n")
                    if context_intervals is None or context_intervals == []:
                        context_intervals = [(0, len(lines))]
                    file_loc_intervals[pred_file] = context_intervals

            found_edit_modules_loc = [[] for _ in range(len(pred_files))]
            for fname, loc_intervals in file_loc_intervals.items():
                i = pred_files.index(fname)
                path = fname.split('/')
                s = deepcopy(structure)
                skip = False
                for p in path:
                    if p in s:
                        s = s[p]
                    else: 
                        skip = True
                if skip:
                    continue
                for loc_interval in loc_intervals:
                    start_line = loc_interval[0]
                    end_line = loc_interval[1]
                    cur_module_end_line = None
                    for line in range(start_line, end_line+1):
                        if cur_module_end_line and line <= cur_module_end_line:
                            continue
                        
                        module, cur_module_end_line = get_module_from_line_number(line, fname, structure)
                        if not module:
                            found_edit_modules_loc[i].append(f"line: {line}")
                            continue

                        if module not in found_edit_modules_loc[i]:
                            found_edit_modules_loc[i].append(module)
                        
            found_edit_modules = [['\n'.join(modules)] for modules in found_edit_modules_loc]
        else:
            found_edit_modules = get_edit_modules_from_file_to_dict(pred_files, file_to_edit_locs, structure)
        
        REPO_GRAPH_LOC = os.environ.get("REPO_GRAPH_LOC")
        code_graph = pickle.load(
            open(f"{REPO_GRAPH_LOC}/graph/{instance_id}.pkl", "rb")
        )
        graph_tags = json.load(
            open(f"{REPO_GRAPH_LOC}/tags/tags_{instance_id}.json", "r")
        )

        one_hop_dependencies = construct_one_hop_dependencies(
            found_edit_modules,
            code_graph,
            graph_tags,
            structure,
        )

        # save the located modules & one_hop_dependencies
        loc['found_edit_modules'] = found_edit_modules
        loc['one_hop_dependencies'] = one_hop_dependencies
        with open(output_file_name, "a") as file:
            file.write(json.dumps(loc) + '\n')
        # break

def run_converte_loc_to_modules(rank, args, loc_queue, log_queue, output_file_lock):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")
    
    while True:
        try:
            loc = loc_queue.get_nowait()
        except Empty:
            break
            
        instance_id = loc["instance_id"]

        if loc["found_files"] == [[]]:
            logger.info(f"{instance_id}'s found_files is emptry.")
            with output_file_lock:
                save_data_to_file(loc, args.output_file)
        
        else:
            if args.dataset == 'princeton-nlp/SWE-bench' or args.loc_bench:
                instance_data = {
                    'instance_id': instance_id,
                    'repo': loc['meta_data']['repo'],
                    'base_commit': loc['meta_data']['base_commit'],
                    'problem_statement': loc['meta_data']['problem_statement'],
                    'patch': loc['meta_data']['patch']
                }
                set_current_issue(instance_data=instance_data)
            else:
                set_current_issue(instance_id)
                
            issue_id, bench_data, structure = get_current_issue_data()
                
            # loc["found_files"] = [loc["found_files"]]
            # loc["found_edit_locs"] = [loc["found_edit_locs"]]
            found_edit_modules = []
            for i in range(len(loc["found_files"])):
                pred_files = loc["found_files"][i]
                file_to_edit_locs = dict()
                for j, temp_edit_locs in enumerate(loc["found_edit_locs"][i]):
                    file_to_edit_locs[pred_files[j]] = temp_edit_locs
                            
                # change to modules
                edit_modules = get_edit_modules_from_file_to_dict(pred_files, file_to_edit_locs, structure, keep_whole_class=False)
                found_edit_modules.append(edit_modules)
            
            # write location to file
            loc['found_edit_locs'] = found_edit_modules
            with output_file_lock:
                save_data_to_file(loc, args.output_file)
                
            reset_current_issue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--used_list", type=str, default='selected_ids')
    parser.add_argument(
        "--model", type=str,
        default="openai/gpt-4o-2024-05-13",
        choices=["gpt-4o", "gpt-35-turbo",
                 "azure/gpt-4", "azure/gpt-4o", "azure/gpt-35-turbo",'azure/gpt-4o-mini',
                 "openai/gpt-4o-2024-05-13", "openai/gpt-4o-mini-2024-07-18", "openai/deepseek-v2.5",
                 "deepseek/deepseek-chat", "deepseek-ai/DeepSeek-R1",
                 "litellm_proxy/claude-3-5-sonnet-20241022", "litellm_proxy/gpt-4o-2024-05-13", "litellm_proxy/o3-mini-2025-01-31",
                 "azure/gpt-4o-mini-ft", "azure/gpt-4o-mini-1029-ft",# fine-tuned model
                 "openai/qwen-7B", "openai/qwen-7B-128k", "openai/ft-qwen-7B", "openai/ft-qwen-7B-128k",
                 "openai/qwen-32B", "openai/qwen-32B-128k", "openai/ft-qwen-32B", "openai/ft-qwen-32B-128k",
                 ]
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_attempt_num", type=int, default=1)
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--add_dependencies", action="store_true")
    parser.add_argument("--expand_context", action="store_true")
    parser.add_argument("--merge_file", type=str, default="merged_loc_outputs.jsonl")
    parser.add_argument("--rerank_file", type=str, default="reranked_loc_outputs.jsonl")
    parser.add_argument("--summary_file", type=str, default="summary_loc_outputs.jsonl")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--log_level", type=str, default='INFO')
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--use_example", action="store_true")
    parser.add_argument("--rerun_empty_location", action="store_true")
    parser.add_argument("--use_function_calling", action="store_true")
    parser.add_argument("--inforce_search_keywords", action="store_true")
    
    parser.add_argument("--selected_pipeline", type=str, 
                        default='auto_search',
                        choices=[
                            'auto_search',
                            'simple_localize',
                            'reconstruct_then_localize',
                            'module_loc_then_bug_loc',
                        ])
    parser.add_argument("--ranking_method", type=str, default='majority', 
                        choices=['mrr', 'majority'])
    parser.add_argument("--loc_bench", action="store_true")
    parser.add_argument("--simple_desc", action="store_true")
    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        # filename=f"{args.output_folder}/localize.log",
        level=logging.getLevelName(args.log_level),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"{args.output_folder}/localize.log"),
            logging.StreamHandler()
        ]
    )
    
    if args.localize:
        localize(args)
        
    if args.merge:
        merge(args)


if __name__ == "__main__":

    start_time = time.time()
    main()
    end_time = time.time()

    logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))
