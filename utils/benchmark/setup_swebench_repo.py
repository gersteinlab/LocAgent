import os
from typing import Optional
from datasets import load_dataset
from utils.benchmark.repo import setup_github_repo
import argparse


def load_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test"
):
    data = load_dataset(dataset_name, split=split)
    return {d["instance_id"]: d for d in data}

def load_instance(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
):
    data = load_instances(dataset_name, split=split)
    return data[instance_id]

def setup_swebench_lite_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = f"swe-bench/{repo_dir_name}"
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )

def setup_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = load_instance(instance_id, dataset_name="princeton-nlp/SWE-bench", split="train")
        # instance_data = [data for data in swe_bench_data if data['instance_id']==instance_id][0]

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    # repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = instance_data["repo"]
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_base_dir", type=str, default='/tmp/repos')
    parser.add_argument("--eval_n_limit", type=int, default=1)

    args = parser.parse_args()

    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    if args.eval_n_limit:
        swe_bench_data = swe_bench_data.select(range(args.eval_n_limit))
    
    for instance in swe_bench_data:
        # repo_base_dir = os.path.join(args.repo_base_dir, instance['instance_id'])
        path = setup_swebench_lite_repo(instance_data=instance, repo_base_dir=args.repo_base_dir)
        print(instance['instance_id'], path)
