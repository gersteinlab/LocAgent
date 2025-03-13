import argparse
import json
import os
import pickle
import time
from pathlib import Path
import subprocess
import torch.multiprocessing as mp
import os.path as osp

from datasets import load_dataset

from build_graph import build_graph, add_edges, convert_edges_to_invokes
from build_graph_v2 import build_graph_v2
from utils.benchmark.setup_swebench_repo import setup_swebench_repo


def list_folders(path):
    return [p.name for p in Path(path).iterdir() if p.is_dir()]


def run(rank, repo_queue, repo_path, out_path, version,
        download_repo=False, instance_data=None):
    while True:
        try:
            repo_name = repo_queue.get_nowait()
        except Exception:
            # Queue is empty
            break

        output_file = f'{osp.join(out_path, repo_name)}.pkl'
        if osp.exists(output_file):
            # print(f'[{rank}] {repo_name} already processed, skipping.')
            continue

        if download_repo:
            # get process specific base dir
            repo_base_dir = str(osp.join(repo_path, str(rank)))
            os.makedirs(repo_base_dir, exist_ok=True)
            # clone and check actual repo
            try:
                repo_dir = setup_swebench_repo(instance_data=instance_data[repo_name], repo_base_dir=repo_base_dir)
            except subprocess.CalledProcessError as e:
                print(f'[{rank}] Error checkout commit {repo_name}: {e}')
                continue
        else:
            repo_dir = osp.join(repo_path, repo_name)

        print(f'Start process {repo_name}')
        try:
            if version == 'v1':
                G = build_graph(repo_dir)
                add_edges(G)
                G = convert_edges_to_invokes(G)
            elif version == 'v2':
                G = build_graph_v2(repo_dir)
            elif version == 'v2.3':
                G = build_graph_v2(repo_dir, global_import=True)
            else:
                raise NotImplementedError(f"Version '{version}' is not implemented.")

            with open(output_file, 'wb') as f:
                pickle.dump(G, f)
            print(f'[{rank}] Processed {repo_name}')
        except Exception as e:
            print(f'[{rank}] Error processing {repo_name}: {e}')


if __name__ == '__main__':
    DEFAULT_REPO_PATH = 'DATA/repo/'
    DEFAULT_TRAINING_REPO_PATH = 'DATA/repo_train/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v2.3')
    parser.add_argument('--repo_path', type=str, default=DEFAULT_REPO_PATH)
    parser.add_argument('--instance_id_path', type=str, default='')
    parser.add_argument('--out_path', type=str, default='')
    parser.add_argument('--num_processes', type=int, default=30)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--loc_bench', action='store_true')
    args = parser.parse_args()

    assert args.version in ['v1', 'v2', 'v2.3']

    if args.out_path == '':
        if args.version == 'v1':
            args.out_path = 'DATA/dependency-graph/'
        else:
            args.out_path = f'DATA/dependency_graph_{args.version}/'
    os.makedirs(args.out_path, exist_ok=True)
        
    # load repo instance id and instance_data
    if args.training or args.loc_bench:
        assert args.instance_id_path != ''

        is_download_repo = True
        if args.repo_path == DEFAULT_REPO_PATH:
            args.repo_path = DEFAULT_TRAINING_REPO_PATH
        selected_instance_data = {}
        
        if args.training:
            with open(args.instance_id_path, 'r') as f:
                repo_folders = json.loads(f.read())

            swe_bench_data = load_dataset("princeton-nlp/SWE-bench", split="train")
            for instance in swe_bench_data:
                if instance['instance_id'] in repo_folders:
                    selected_instance_data[instance['instance_id']] = instance
        else:
            repo_folders = []
            with open(args.instance_id_path, 'r') as f:
                for line in f:
                    instance = json.loads(line)
                    repo_folders.append(instance['instance_id'])
                    selected_instance_data[instance['instance_id']] = instance
    else:
        if args.instance_id_path:
            with open(args.instance_id_path, 'r') as f:
                repo_folders = json.loads(f.read())
        else:
            repo_folders = list_folders(args.repo_path)
        selected_instance_data = None
        is_download_repo = False

    os.makedirs(args.repo_path, exist_ok=True)

    # Create a shared queue and add repositories to it
    manager = mp.Manager()
    queue = manager.Queue()
    for repo in repo_folders:
        queue.put(repo)

    start_time = time.time()

    # Start multiprocessing with a global queue
    mp.spawn(
        run,
        nprocs=args.num_processes,
        args=(queue, args.repo_path, args.out_path, args.version,
              is_download_repo, selected_instance_data),
        join=True
    )

    end_time = time.time()
    print(f'Total Execution time = {end_time - start_time:.3f}s')
