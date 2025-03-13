from plugins.location_tools.utils.get_repo_structure.get_repo_structure import (
    get_project_structure_from_scratch
)
from agentless.util.utils import load_jsonl
from datasets import load_dataset
from tqdm import tqdm
import torch.multiprocessing as mp
import time
import argparse
import json
import subprocess
import uuid
import os
import logging
import logging.handlers
from queue import Empty


def get_project_structure(instance_data, repo_output_dir, repo_playground='repo_playground'):
    instance_id = instance_data['instance_id']
    try:
        d = get_project_structure_from_scratch(
            instance_data["repo"], instance_data["base_commit"], instance_id, repo_playground
        )
        if not d:
            return False
        # instance_id = instance_data['instance_id']
        repo_structure_dir = os.path.join(repo_output_dir, f'{instance_id}.json')
        with open(repo_structure_dir, 'w') as f:
            json.dump(d, f, indent=4)
        return True
    except subprocess.CalledProcessError as e:
        print("Exception on process, rc=", e.returncode, "output=", e.output)
        return False
    

def run_get_project_structure(rank, queue, log_queue, output_file_lock,
                              repo_output_dir, repo_playground, log_file
                              ):
    
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName("INFO"))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")
    
    while True:
        try:
            instance = queue.get_nowait()
        except Empty:
            break
        
        start_time = time.time()
        # Generate a temperary folder and add uuid to avoid collision
        repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))
        # assert playground doesn't exist
        assert not os.path.exists(repo_playground), f"{repo_playground} already exists"
        # create playground
        os.makedirs(repo_playground)
        succuess = get_project_structure(instance, repo_output_dir, repo_playground)
        
        # clean up
        subprocess.run(
            ["rm", "-rf", f"{repo_playground}"], check=True
        )
        end_time = time.time()
        with output_file_lock:
            with open(os.path.join(repo_output_dir, log_file), "a") as f:
                f.write("{:.2f}".format((end_time - start_time)) + "\n")


def main(args):
    # swe_bench_data = load_dataset("princeton-nlp/SWE-bench", split="train")
    # selected_list_file = '/home/ubuntu/auto-search-agent/scripts/notebooks/fine-tune/data/selected_instances_20241122.json'
    assert os.path.exists(args.dataset_file)
    bench_data = load_jsonl(args.dataset_file)
    bench_data = bench_data[:args.eval_n_limit]
    # 先检查已经生成的
    generated_instances = []
    for filename in os.listdir(args.repo_output_dir):
        # instance_id = filename.split('.')[0]
        instance_id = filename[:-5]
        generated_instances.append(instance_id)
    
    manager = mp.Manager()
    queue = manager.Queue()
    output_file_lock = manager.Lock()
    num_tasks = 0
    for data in bench_data:
        if data['instance_id'] not in generated_instances:
            queue.put(data)
            num_tasks += 1
    
    log_queue = manager.Queue()
    queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()
    mp.spawn(
        run_get_project_structure,
        nprocs=min(num_tasks, args.num_processes) if args.num_processes > 0 else num_tasks,
        args=(queue, log_queue, output_file_lock,
              args.repo_output_dir, args.repo_playground, args.log_file,
            ),
        join=True
    )
    queue_listener.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_playground", type=str, default="playground/repo_struc")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, default="time.log")
    parser.add_argument("--repo_output_dir", type=str, required=True)
    parser.add_argument("--eval_n_limit", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=1)
    
    args = parser.parse_args()
    
    start_time = time.time()
    main(args)
    end_time = time.time()

    logging.info("Total time: {:.4f} min".format((end_time - start_time)/60))