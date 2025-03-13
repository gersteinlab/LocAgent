# Contributing to auto-search-agent

We welcome contributions from everyone to help improve and expand auto-search-agent. This document outlines the process for contributing to the project.

## Table of Contents
1. [Environment Setup](#environment-setup)

## Environment Setup

To contribute to *auto-search-agent*, follow these steps to set up your development environment:

1. Clone the repository:
   ```
   git clone git@github.com:gersteinlab/swebench.git
   cd swebench
   ```
2. Create a Conda environment:
   ```
   conda create -n swebench python=3.11
   conda activate swebench
   ```
3. Install the project in editable mode with development dependencies:
   ```
   python3 -m pip install --upgrade pip
   pip install -e .
   ```

4. Set the environment variable.
   - add `key` in `scripts/env/set_env.sh`
   ```
   # open-hands key
   # use lite-llm proxy model
   export OPENAI_API_KEY="sk-123..."
   export OPENAI_API_BASE="https://XXXXX"

   # use deep-seek model
   export DEEPSEEK_API_KEY="sk-123..."

   # azure: openai embedding
   export AZURE_OPENAI_API_KEY_EMBED="123.."
   export AZURE_OPENAI_ENDPOINT_EMBED="https://XXXXX"
   ```

   - Move the cache data and then set the environment variable.
     - PROJECT_FILE_LOC='/home/ubuntu/auto-search-agent/get_repo_structure/repo_structures/SWE-bench'
     - DEPENDENCY_GRAPH_LOC='/home/ubuntu/auto-search-agent/graph_encoder/DATA/training_dp_graph_v2.1'
     - INDEX_STORE_LOC='/home/ubuntu/auto-search-agent/index_data/SWE-bench/20241016-text-embedding-3-small'
     - GT_LOCS_FILE='/home/ubuntu/auto-search-agent/evaluation/gt_data/SWE-bench/gt_locs_data_20241022.json'

1. Run the script
   ```
   cd auto-search-agent
   bash scripts/deepseek-v2.5/run.sh
   ```
