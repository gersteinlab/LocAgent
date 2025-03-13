import argparse
import ast
import json
import os
import subprocess
import uuid

import pandas as pd
from tqdm import tqdm
import time
# import logging

# logger = logging.getLogger(__name__)

repo_to_top_folder = {
    "django/django": "django",
    "sphinx-doc/sphinx": "sphinx",
    "scikit-learn/scikit-learn": "scikit-learn",
    "sympy/sympy": "sympy",
    "pytest-dev/pytest": "pytest",
    "matplotlib/matplotlib": "matplotlib",
    "astropy/astropy": "astropy",
    "pydata/xarray": "xarray",
    "mwaskom/seaborn": "seaborn",
    "psf/requests": "requests",
    "pylint-dev/pylint": "pylint",
    "pallets/flask": "flask",
}


def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
        return True
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return True

def maybe_clone(repo_name, repo_playground):
    repo_url = f"https://github.com/{repo_name}.git"
    repo_dir = f"{repo_playground}/{get_repo_dir_name(repo_name)}"

    if not os.path.exists(f"{repo_dir}/.git"):
        while True:
            try:
                # logger.info(f"Cloning repo '{repo_url}'")
                # Clone the repo if the directory doesn't exist
                result = subprocess.run(
                    ["git", "clone", repo_url, repo_dir],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                if result.returncode == 0:
                    # logger.info(f"Repo '{repo_url}' was cloned to '{repo_dir}'")
                    success = True
                    break
                else:
                    # logger.info(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")
                    success = False
                    break
                    # raise ValueError(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")
            except subprocess.CalledProcessError as e:
                print(e)
                success = False
                time.sleep(10)
    else:
        success = True
            
    return success

def clone_repo(repo_name, repo_playground):
    try:

        print(
            f"Cloning repository from https://github.com/{repo_name}.git to {repo_playground}/{get_repo_dir_name(repo_name)}..."
        )
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                f"{repo_playground}/{get_repo_dir_name(repo_name)}",
            ],
            check=True,
        )
        print("Repository cloned successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def get_repo_dir_name(repo: str):
    return repo.replace("/", "_")

def get_project_structure_from_scratch(
    repo_name, commit_id, instance_id, repo_playground
):

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    success = maybe_clone(repo_name, repo_playground)
    if not success: return None
    
    success = checkout_commit(f"{repo_playground}/{get_repo_dir_name(repo_name)}", commit_id)
    if not success: return None

    structure = create_structure(f"{repo_playground}/{get_repo_dir_name(repo_name)}")
    if not structure: return None
    # clean up
    subprocess.run(
        ["rm", "-rf", f"{repo_playground}/{get_repo_dir_name(repo_name)}"], check=True
    )
    d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return d


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) or isinstance(
            node, ast.AsyncFunctionDef
        ): # include `AsyncFunctionDef`
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                if os.path.islink(file_path):
                    continue
                
                class_info, function_names, file_lines = parse_python_file(file_path)
                if not class_info and not function_names and not file_lines:
                    # parse error
                    # return None
                    continue
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure
