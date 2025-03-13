Note1: cd to this folder before executing the following commands 

Note2: include project root: `export PYTHONPATH=/home/gangda/workspace/auto-search-agent:$PYTHONPATH`

- Construct dependency-graph for all the repositories
    ```
    python dependency_graph/batch_build_graph.py \
        --repo_path parent_dir_for_all_repositories \
        --out_path generated_graphs_path \
        --num_processes 30
    ```
  
- Download and checking repos 
  ```
    python dependency_graph/batch_build_graph.py \
        --repo_path parent_dir_for_all_repositories \
        --instance_id_path selected_instance_id_list_path \
        --out_path generated_graphs_path \
        --num_processes 30
        --download_repo
    ```