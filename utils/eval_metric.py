import json
import os
import sys
import pandas as pd
from agentless.util.utils import load_jsonl, load_json
from typing import Optional
from torch import Tensor
import torch
import pickle
DEPENDENCY_GRAPH_LOC = os.environ.get("DEPENDENCY_GRAPH_LOC")
from graph_encoder.dependency_graph import RepoEntitySearcher


filtered_instances=['pytest-dev__pytest-5227',
 'sympy__sympy-15345',
 'sympy__sympy-21614',
 'scikit-learn__scikit-learn-13439',
 'sympy__sympy-11400',
 'sympy__sympy-19487',
 'sympy__sympy-15308',
 'django__django-12915',
 'sympy__sympy-20590',
 'sympy__sympy-17022',
 'django__django-11099',
 'django__django-13220',
 'django__django-11964',
 'matplotlib__matplotlib-25332',
 'django__django-10914',
 'django__django-14915',
 'django__django-11049',
 'django__django-11564',
 'sympy__sympy-17655',
 'sympy__sympy-16106',
 'sympy__sympy-12171',
 'django__django-15400',
 'django__django-14411',
 'sympy__sympy-21055',
 'django__django-15213',
 'django__django-15902',
 
 'tobymao__sqlglot-4477',
 'pyodide__pyodide-1215',
 'django__django-17811',
 'tobymao__sqlglot-4497',
 'optuna__optuna-5778',
 'sagemath__sage-36814',
 'tobymao__sqlglot-4457',
 'tobymao__sqlglot-4505',
 'tobymao__sqlglot-4430',
 'raft-tech__TANF-app-3234',
'AllenNeuralDynamics__foraging-behavior-browser-106',
'Australian-Text-Analytics-Platform__atap-corpus-slicer-3',
 ]

def _dcg(target: Tensor) -> Tensor:
    batch_size, k = target.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
    return (target / torch.log2(rank_positions + 1)).sum(dim=-1)

def div_no_nan(a: Tensor, b: Tensor, na_value: Optional[float] = 0.) -> Tensor:
    return (a / b).nan_to_num_(nan=na_value, posinf=na_value, neginf=na_value)

def normalized_dcg(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]
    ideal_target = ideal_target[:, :k]
    return div_no_nan(_dcg(pred_target), _dcg(ideal_target)).mean(0)


def recall_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    relevant = (pred_target == 1).sum(dim=-1)  # 计算预测中相关文档的个数
    total_relevant = (ideal_target == 1).sum(dim=-1)  # 计算所有相关文档的个数
    recall = div_no_nan(relevant, total_relevant, na_value=0.)  # 计算 Recall@k
    return recall.mean(0)


def acc_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    ideal_target = ideal_target[:, :k]
    
    relevant = (pred_target == 1).sum(dim=-1)  # 计算预测中相关文档的个数
    total_relevant = (ideal_target == 1).sum(dim=-1)  # 计算所有相关文档的个数

    comparison = relevant == total_relevant
    return comparison.sum()/relevant.shape[0]


def precision_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    relevant = (pred_target == 1).sum(dim=-1)  # 计算预测中相关文档的个数
    precision = relevant / k  # 计算 Precision@k
    return precision.mean(0)


def average_precision_at_k(pred_target: Tensor, ideal_target: Tensor, k: Optional[int] = None) -> Tensor:
    batch_size, k_val = pred_target.shape
    pred_target = pred_target[:, :k]  # 只考虑前 k 个预测结果
    ideal_target = ideal_target[:, :k]
    
    precisions = []
    for i in range(batch_size):
        ap = 0.0
        relevant_count = 0
        for j in range(k):
            if pred_target[i, j] == 1:  # 如果是相关文档
                relevant_count += 1
                ap += relevant_count / (j + 1)  # 计算 Precision@j
        # if relevant_count > 0:
        ap = ap/k
        # print(ap)
        precisions.append(ap)
    
    return torch.tensor(precisions).mean()


def load_gt_dict(gt_file, level):
    gt_datas = load_jsonl(gt_file)
    # gt_data = [data for data in gt_datas if data['instance_id']==instance_id][0]
    
    gt_dict = {}
    for gt_data in gt_datas:
        gt_locs = []
        instance_id = gt_data['instance_id']
        file_changes = gt_data['file_changes']
        for file_change in file_changes:
            if level == 'file':
                gt_locs.append(file_change['file'])
            elif level == 'module':
                changes = file_change['changes']
                if 'edited_modules' in changes:
                    gt_locs.extend(changes['edited_modules'])
            elif level == 'function':
                changes = file_change['changes']
                if 'edited_entities' in changes:
                    gt_locs.extend(changes['edited_entities'])
                    # filtered_gt_entities = []
                    # edited_entities = changes['edited_entities']
                    # G = pickle.load(
                    #     open(f"{DEPENDENCY_GRAPH_LOC}/{instance_id}.pkl", "rb")
                    # )
                    # searcher = RepoEntitySearcher(G)
                    # for gt_entity in edited_entities:
                    #     if gt_entity.endswith('.__init__'):
                    #         gt_entity = gt_entity[:(len(gt_entity)-len('.__init__'))]
                    #         filtered_gt_entities.append(gt_entity)
                    #         continue
                    #     ndata = searcher.get_node_data([gt_entity])[0]
                    #     if ndata['type'] == 'function':
                    #         filtered_gt_entities.append(gt_entity)
                    # gt_locs.extend(filtered_gt_entities)
        # return gt_locs
        gt_dict[gt_data['instance_id']] = gt_locs
    return gt_dict

import re
def extract_file_path(changed_funcs):
    for k, v in changed_funcs.items():
        changed_files = []
        seen_files = set()
        for vv in v:
            match = re.match(r"(.+\.py)(/.*)?", vv)
            if match:
                if match.group(1) not in seen_files:
                    changed_files.append(match.group(1))
                    seen_files.add(changed_files[-1])
            else:
                import pdb;pdb.set_trace()  
        changed_funcs[k] = changed_files
    
    return changed_funcs


def convert_solutions_dict(dataset, key = 'model_patch'):
    return {elem['instance_id']: elem[key] for elem in dataset}


METRIC_FUNC = {
    'ndcg': normalized_dcg,
    'recall': recall_at_k,
    'acc': acc_at_k,
    'precision': precision_at_k,
    'map': average_precision_at_k
}
METRIC_NAME = {
    'ndcg': 'NDCG',
    'recall': 'Recall',
    'acc': 'Acc',
    'precision': 'P',
    'map': 'MAP'
}
def cal_metrics(gt_file, loc_file, key,
                level,
                k_values, # < 100
                metrics=['acc', 'ndcg', 'precision', 'recall', 'map'],
                filter_list=filtered_instances,
                selected_list=None,
                merge_init = True,
                ):
    assert key in ['found_files', 'found_modules', 'found_entities', 'docs']
    
    max_k = max(k_values)
    # loc_output = load_jsonl(loc_file)
    gt_dict = load_gt_dict(gt_file, level)
    if key == 'docs' and level == 'file':
        pred_dict = extract_file_path(convert_solutions_dict(load_jsonl(loc_file), key='docs'))
    elif key == 'docs':
        pred_dict = convert_solutions_dict(load_jsonl(loc_file), key='docs')
        for ins in pred_dict:
            pred_funcs = pred_dict[ins]
            pred_modules = []
            for i, pl in enumerate(pred_funcs):
                fle, func_n = pl.split('.py/')
                if level == 'function':
                    if func_n.endswith('.__init__'):
                        func_n = func_n[:(len(func_n)-len('.__init__'))]
                    pred_funcs[i] = f'{fle}.py:{func_n.strip('/').replace('/', '.')}'
                elif level == 'module':
                    module_name = f'{fle}.py:{func_n.strip('/').split('/')[0]}'
                    if module_name not in pred_modules:
                        pred_modules.append(module_name)
                    pred_dict[ins] = pred_modules
    else:
        pred_dict = convert_solutions_dict(load_jsonl(loc_file), key=key)
        
    _gt_labels = []
    _pred_labels = []
    
    # for loc in loc_output:
    for instance_id in gt_dict.keys():
        # instance_id = loc['instance_id']
        if filter_list and instance_id in filter_list: continue # filter
        if selected_list and instance_id not in selected_list: continue
        if not gt_dict[instance_id]: continue
        
        if instance_id not in pred_dict:
            pred_locs = []
        else:
            pred_locs = pred_dict[instance_id][: max_k]
                
        gt_labels = [0 for _ in range(max_k)]
        pred_labels = [0 for _ in range(max_k)]

        for i in range(len(gt_dict[instance_id])):
            if i < max_k:
                gt_labels[i] = 1
        
        for i, l in enumerate(pred_locs):
            if l in gt_dict[instance_id]:
                pred_labels[i] = 1
                
        _gt_labels.append(gt_labels)
        _pred_labels.append(pred_labels)
    
    _pred_target = torch.tensor(_pred_labels)
    _ideal_target = torch.tensor(_gt_labels)
    
    result = {}
    for metric in metrics:
        assert metric in METRIC_FUNC.keys()
        
        metric_func = METRIC_FUNC[metric]
        name = METRIC_NAME[metric]
        for k in k_values:
            value = metric_func(_pred_target, _ideal_target, k=k)
            result[f'{name}@{k}'] = round(value.item(), 4)
            
    return result


def cal_metrics_w_file(gt_file, loc_file, level2key_dict, selected_list=None, k_values_list=None):
    if not k_values_list:
        k_values_list = [
            [1, 2, 3, 5],
            [3, 5, 10],
            [3, 5, 10]
        ]
    file_res = cal_metrics(gt_file, loc_file, 
                            level2key_dict['file'], level='file', k_values=k_values_list[0],
                            selected_list=selected_list)
    module_res = cal_metrics(gt_file, loc_file, 
                            level2key_dict['module'], level='module', k_values=k_values_list[1],
                            selected_list=selected_list)
    function_res = cal_metrics(gt_file, loc_file, 
                            level2key_dict['function'], level='function', k_values=k_values_list[2],
                            selected_list=selected_list)

    all_df = pd.concat([pd.DataFrame(res, index=[0])
                          for res in [file_res, module_res, function_res]], 
                        axis=1, 
                        keys=['file', 'module', 'function'])
    return all_df