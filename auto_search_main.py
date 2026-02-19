import argparse
import os
import json
import logging
import logging.handlers
import time
from queue import Empty
from typing import List
from copy import deepcopy
from util.config import load_config

from util.runtime.execute_ipython import execute_ipython
from util.runtime import function_calling
from util.actions.action_parser import ResponseParser
from util.actions.action import ActionType
from util.prompts.prompt import PromptManager
from util.prompts import general_prompt
from util.prompts.pipelines import (
    simple_localize_pipeline as simple_loc,
    auto_search_prompt as auto_search,
)
from util.cost_analysis import calc_cost
from util.utils import *
from util.process_output import (
    parse_raw_loc_output,
    get_loc_results_from_raw_outputs,
    merge_sample_locations,
)
from plugins import LocationToolsRequirement
from plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
)
import litellm
from litellm import Message as LiteLLMMessage
from openai import APITimeoutError


from time import sleep
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import torch.multiprocessing as mp
from util.runtime.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
    STOP_WORDS as NON_FNCALL_STOP_WORDS
)
# litellm.set_verbose=True
# os.environ['LITELLM_LOG'] = 'DEBUG


_LLM_CONFIG = None


def _get_llm_config() -> dict:
    global _LLM_CONFIG
    if _LLM_CONFIG is None:
        _LLM_CONFIG, _ = load_config()
    return _LLM_CONFIG


def _is_nim_model(model_name: str) -> bool:
    return model_name.startswith('nvidia_nim/')


def _resolve_api_config(model_name: str, config: dict):
    api_base = None
    api_key = None
    if _is_nim_model(model_name):
        api_base = config.get('nim_api_base', 'https://integrate.api.nvidia.com/v1')
        api_key = os.getenv(config.get('api_key_env', 'NVIDIA_API_KEY'))
    return api_base, api_key


def _resolve_fallback_config(config: dict):
    fallback_model = config.get('fallback_model')
    fallback_base = os.getenv('OPENAI_API_BASE')
    fallback_key = os.getenv('OPENAI_API_KEY')
    if fallback_model and fallback_key:
        return fallback_model, fallback_base, fallback_key
    return None, None, None


def _completion_with_retry(model_name: str, **kwargs):
    config = _get_llm_config()
    max_retries = int(config.get('max_retries', 3))
    backoff = float(config.get('retry_backoff_sec', 2))
    api_base, api_key = _resolve_api_config(model_name, config)

    last_error = None
    for attempt in range(max_retries):
        try:
            return litellm.completion(model=model_name, api_base=api_base, api_key=api_key, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                sleep(backoff * (2 ** attempt))
                continue

    fallback_model, fallback_base, fallback_key = _resolve_fallback_config(config)
    if _is_nim_model(model_name) and fallback_model:
        return litellm.completion(model=fallback_model, api_base=fallback_base, api_key=fallback_key, **kwargs)

    if last_error:
        raise last_error
    raise RuntimeError('LLM completion failed without an exception.')


def get_task_instruction(instance: dict, task: str = 'auto_search', include_pr=False, include_hint=False):
    output_format = None
    instruction = ""
    
    # for auto-search pipeline
    if task.strip() == 'auto_search':
        task_description = auto_search.TASK_INSTRUECTION.format(
            package_name=instance['instance_id'].split('_')[0]
        )
    
    elif task.strip() == 'simple_localize':
        task_description = simple_loc.SEARCH_LOC_TASK_INSTRUCTION
        output_format = simple_loc.OUTPUT_FORMAT_LOC
        
    else:
        return None

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
                        temp=1.0,
                        max_iteration_num=20,
                        use_function_calling=True):
    if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower() 
    #             #   or model_name=='azure/gpt-4o' 
    #             #   or model_name == 'litellm_proxy/o3-mini-2025-01-31'
                ):
        use_function_calling = False
        
    # for LLM which do not support function calling
    if not use_function_calling:
        # 转换message
        messages = convert_fncall_messages_to_non_fncall_messages(messages, tools, add_in_context_learning_example=False)
            
    # code_history = []
    parser = ResponseParser()
    if not traj_data:
        traj_msgs = messages.copy()
        prompt_tokens = 0
        completion_tokens = 0
    else:
        # continue from last traj
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
                response = _completion_with_retry(
                    model_name,
                    temperature=temp, top_p=0.8, repetition_penalty=1.05,
                    messages=messages,
                    stop=NON_FNCALL_STOP_WORDS
                )
            elif tools:
                response = _completion_with_retry(
                    model_name,
                    tools=tools,
                    messages=messages,
                    temperature=temp,
                    # stop=['</execute_ipython>'], #</finish>',
                )
            else:
                response = _completion_with_retry(
                    model_name,
                    messages=messages,
                    temperature=temp,
                    stop=['</execute_ipython>'], #</finish>',
                )
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
        if tools and ('hosted_vllm' in model_name or 'qwen' in model_name.lower()
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
                logging.info('='*15)
                logging.info("\nFinal Response:=\n" + final_output)
                finish = True # break
            elif action.action_type == ActionType.MESSAGE:
                logging.debug("thought:\n" + action.content)
                # check if enough
                messages.append({"role": "user", "content": fake_user_msg})
                traj_msgs.append({"role": "user", "content": fake_user_msg})
                # continue
            elif action.action_type == ActionType.RUN_IPYTHON:
                ipython_code = action.code.strip('`')
                logging.info(f"Executing code:\n```\n{ipython_code}\n```")
                function_response = execute_ipython(ipython_code)
                try:
                    function_response = eval(function_response)
                except SyntaxError:
                    function_response = function_response
                if not isinstance(function_response, str):
                    function_response = str(function_response)
                
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


def run_localize(rank, args, bug_queue, log_queue, output_file_lock, traj_file_lock):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.log_level))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")

    while True:
        try:
            bug = bug_queue.get_nowait()
        except Empty:
            break

        instance_id = bug["instance_id"]
        prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'util/prompts'),
            agent_skills_docs=LocationToolsRequirement.documentation,
        )

        logger.info("=" * 60)
        logger.info(f"==== rank {rank} setup localize {instance_id} ====")
        set_current_issue(instance_data=bug, rank=rank)

        # loc result
        raw_output_loc = []
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
                        
                    if args.use_example:
                        messages.append({
                            "role": "user",
                            "content": prompt_manager.initial_user_message
                        })

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
                            codeact_enable_tree_structure_traverser=True,
                            simple_desc = args.simple_desc,
                        )
                    process = ctx.Process(target=auto_search_process, kwargs={
                        'result_queue': result_queue,
                        'model_name': args.model,
                        'messages': messages,
                        'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC,
                        'temp': 1,
                        'tools': tools,
                        'use_function_calling': args.use_function_calling,
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

                loc_end_time = time.time()
                if not loc_result:
                    continue # empty result

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
                    "found_modules": [[]],
                    "found_entities": [[]],
                    "raw_output_loc": raw_output_loc,
                    "meta_data": {
                        'repo': bug['repo'],
                        'base_commit': bug['base_commit'],
                        'problem_statement': bug['problem_statement'],
                        'patch': bug['patch'],
                        # 'gt_file_changes': gt_file_changes
                    }
                }
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)
        else:
            # process multiple loc outputs
            logger.info(f"==== localizing {instance_id} succeed, process multiple loc outputs ====")

            # all_valid_files = get_all_valid_files()
            all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(
                instance_id, raw_output_loc
            )
            
            loc_res = {
                "instance_id": instance_id,
                "found_files": all_found_files,
                "found_modules": all_found_modules,
                "found_entities": all_found_entities,
                "raw_output_loc": raw_output_loc,
                "meta_data": {
                    'repo': bug['repo'],
                    'base_commit': bug['base_commit'],
                    'problem_statement': bug['problem_statement'],
                    'patch': bug['patch'],
                    # 'gt_file_changes': gt_file_changes
                }
            }
            
            with output_file_lock:
                append_to_jsonl(loc_res, args.output_file)

            cost = calc_cost(args.model, total_prompt_tokens, total_completion_tokens)
            loc_res['usage'] = {'cost($)': f'{round(cost, 5)}', 'prompt_tokens': total_prompt_tokens,
                                'completion_tokens': total_completion_tokens}
            loc_res['loc_trajs'] = loc_trajs
            traj_file = os.path.join(args.output_folder, 'loc_trajs.jsonl')
            with traj_file_lock:
                append_to_jsonl(loc_res, traj_file)

        reset_current_issue()
