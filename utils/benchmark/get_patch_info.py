import os
import re
import collections
from datasets import load_dataset
from tqdm import tqdm
import unidiff
from unidiff.errors import UnidiffParseError
import json

def get_edited_file(patch_content):
    # Extract the file path from the patch content
    file_path_match = re.search(r'--- a/(.*?)\n\+\+\+ b/\1', patch_content)
    if not file_path_match:
        return None
        # raise ValueError("File path not found in patch content")

    file_path = file_path_match.group(1)
    return file_path

def get_edited_files(patch_content):
    # Extract all file paths from the patch content
    # file_paths = re.findall(r'--- a/(.*?)\n\+\+\+ b/\1', patch_content)
    file_paths = re.findall(r'diff --git a/(.*?) b/\1', patch_content)
    
    if not file_paths:
        return None
        # raise ValueError("No file paths found in patch content")
    
    return file_paths


def get_oracle_filenames(patch):
    """
    Returns the filenames that are changed in the patch
    """
    source_files = {
        patch_file.source_file.split("a/", 1)[-1]
        for patch_file in unidiff.PatchSet(patch)
    }
    gold_docs = set()
    for source_file in source_files:
        gold_docs.add(source_file)
    return gold_docs

def get_edited_lines(diff_text):
    included_lines = []
    included_ranges = []
    # edited_lines = []
    added_lines = []
    removed_lines = []
    
    # Split the diff text into chunks by each hunk
    hunks = diff_text.strip().split('\n@@ ')

    for hunk in hunks:
        if not hunk.strip():
            continue
        
        # Parse the hunk header to get line numbers and context
        header, *changes = hunk.split('\n')
        match = re.match(r'-(\d+),(\d+) \+(\d+),(\d+)', header)
        if not match:
            continue

        start_line_orig = int(match.group(1))
        num_lines_orig = int(match.group(2))
        start_line_new = int(match.group(3))
        num_lines_new = int(match.group(4))
        included_ranges.append((start_line_orig, start_line_orig+num_lines_orig-1))
        included_lines += list(range(start_line_orig, start_line_orig+num_lines_orig))

        current_line_orig = start_line_orig - 1
        current_line_new = start_line_orig - 1
        
        add_action = False
        for line in changes:
            # if line.startswith('-') or line.startswith('+'):
            if line.startswith('+'):
                if not add_action:
                    current_line_new += 1
                    added_lines.append(current_line_new)
                add_action = True
            elif line.startswith('-'):
                if add_action:
                    current_line_new = current_line_orig
                    add_action = False
                current_line_orig += 1
                removed_lines.append(current_line_orig)
            else:
                if add_action:
                    current_line_new = current_line_orig
                    add_action = False
                current_line_orig += 1
                current_line_new += 1

    added_lines = sorted(list(set(included_lines) & set(added_lines)))
    removed_lines = sorted(list(set(removed_lines)))
    edited_lines = sorted(list((set(added_lines) | set(removed_lines))))
    return edited_lines, included_ranges

def split_patch(patch_text):
    # Regex to match the beginning of each diff in a unified diff file
    diff_start_pattern = r'^(diff --git a/.*? b/.*?$)'
    
    # Split the patch into individual diffs using the regex
    diff_parts = re.split(diff_start_pattern, patch_text, flags=re.MULTILINE)
    
    # Combine each header with its corresponding content, ignoring the first empty split part
    diff_list = [f"{header.strip()}\n{content.lstrip()}" for header, content in zip(diff_parts[1::2], diff_parts[2::2])]

    return diff_list

def analyze_swe_dataset(dataset, max_edit_file_num=5, ignore_error=True, selected_list=None, output_file=None):
    file_num_dist = collections.defaultdict(list)
    repo_dist = collections.defaultdict(list)
    gt_instances = dict()
    for data in tqdm(dataset):
        try:
            if selected_list and data['instance_id'] not in selected_list:
                continue
            filenames = get_oracle_filenames(data['patch'])
            if not len(filenames):
                continue
            if max_edit_file_num and len(filenames) > max_edit_file_num:
                continue
            pass_flag = False
            for file in filenames:
                if not file.endswith('.py'):
                    pass_flag = True
            if pass_flag:
                continue
            file_num_dist[len(filenames)].append(data['instance_id'])
            repo_dist[data['repo']].append(data['instance_id'])

            # 分出不同file下的patch，各自计算edit_lines/edit_ranges
            split_diffs = split_patch(data['patch'])
            patch_files = dict()
            for diff in split_diffs:
                edited_file = list(get_oracle_filenames(diff))
                assert len(edited_file) == 1
                edited_file = edited_file[0]
                edited_lines, included_ranges= get_edited_lines(diff)
                if edited_file not in patch_files:
                    patch_files[edited_file] = {
                        'edited_lines': [],
                        'included_ranges': []
                    }
                patch_files[edited_file]['edited_lines'] += edited_lines
                patch_files[edited_file]['included_ranges'] += included_ranges

            gt_instances[data['instance_id']] = {'repo': data['repo'],
                                                 'base_commit': data['base_commit'],
                                                'problem_statement': data['problem_statement'],
                                                'patch': data['patch'], 
                                                'patch_files': patch_files, 
                                                }
        except UnidiffParseError:
            if not ignore_error:
                print("Incomplete patch, skip:", data['instance_id']) # repo: pandas-dev
            continue
        except:
            if not ignore_error:
                print('Error:', data['instance_id'])

    if output_file:
        with open(output_file, 'w') as file:
            json.dump(gt_instances, file)
    
    return (file_num_dist, repo_dist, gt_instances)