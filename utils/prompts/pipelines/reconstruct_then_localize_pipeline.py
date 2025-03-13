
# For fragment tools
RECONSTRUCT_FLOW_TASK="""
# Task:
To resolve real-world GitHub issues effectively, the solution can be divided into two primary stages: localization (identifying relevant files and code snippets) and editing (making necessary code changes). 
Your objective is to localize the specific files, classes or functions declarations, and lines of code that need modification or contain key information to resolve the issue.

# Localization Process:
The localization stage can be broken down further into two key parts:
1. Understanding and Reproducing the Problem: 
    - Analyze the Problem Statement to gather a comprehensive list of all files and modules involved.
    - Identify the sequence of execution described in the problem.
2. Identifying Files and Modules for Modification:
    - Pinpoint the specific file(s) and module(s) that require changes to resolve the issue.

Now, given the following GitHub problem description, the focus is on analyzing the problem and reconstructing the flow of execution. 

## Task 1: Understanding and Reproducing the Problem
The task is divided into three steps:
1. Analyze the issue
    - Goal: Fully understand the problem described in the Problem Statement.
    - Action: Carefully read and interpret the problem description. Identify potential causes, such as logical errors, misconfigurations, or faulty dependencies.
2. Extract Keywords and Search for Code References
    - Goal: Collect ALL the relevant files, classes, functions and variable mentioned in the problem.
    - Action: 
        - Extract ALL the file/class/function/variable names appeared the Problem Statement and any other key words (e.g., the function description or the potential name of the module). 
        - Call retrieval-based functions such as `search_in_repo`, `get_file_structures` or `search_class_structures` to gather more information on these components.
3. Walk through the repository by calling tools and Reconstruct the Execution Flow
    - Goal: Understand the flow of execution by identifying the sequence of function calls and class interactions leading to the issue.
    - Action:
        - Identify the main entry point (e.g., class instantiation or method invocation) that triggers the issue.
        - Trace the sequence of events (function calls, function executions, class instantiation, class inheritance) and note how various parts of the code interact.
        - Use tools such as `get_file_content`, `search_class` or `search_method` to get the full implementation of the modules and to explore files, classes, and methods to fully understand the flow and dependencies within the system.
        - Identify any breakpoints where the issue might arise, based on the reconstructed flow.
Important: Please check carefully that the Execution Flow you generated is related with the problem and don't get bogged down in irrelevant details.

"""

# For all tools
RECONSTRUCT_FLOW_TASK_GENERAL="""
# Task:
To resolve real-world GitHub issues effectively, the solution can be divided into two primary stages: localization (identifying relevant files and code snippets) and editing (making necessary code changes). 
Your objective is to localize the specific files, classes or functions declarations, and lines of code that need modification or contain key information to resolve the issue.

# Localization Process:
The localization stage can be broken down further into two key parts:
1. Understanding and Reproducing the Problem: 
    - Analyze the Problem Statement to identify all relevant files and modules.
    - Trace the described sequence of execution to understand how it leads to the issue.
2. Identifying Files and Modules for Modification:
    - Pinpoint the specific file(s) and module(s) that require changes to resolve the issue.

## Task 1: Understanding and Reproducing the Problem
Now, given the following GitHub problem description, the focus is on analyzing the problem and reconstructing the flow of execution. 
1. Analyze the Issue:
    - Thoroughly review the problem statement to understand the issue.
2. Extract Keywords and Search for Code References:
    - Collect ALL the relevant files, classes, functions and variable mentioned in the Problem Statement to gather more context of the problem.
3. Walk through the repository and Reconstruct the Execution Flow
    - Goal: Understand the flow of execution by identifying the sequence of function calls and class interactions leading to the issue.
    - Action:
        - Identify the main entry point (e.g., class instantiation or method invocation) that triggers the issue.
        - Trace the sequence of events (function calls, class instantiation, class inheritance, etc.) to see how various parts of the code interact.
        - Identify potential breakpoints causing the issue based on the reconstructed flow.
Important: Ensure the reconstructed execution flow is directly related to the problem, avoiding irrelevant details.

"""

# For unified tools
RECONSTRUCT_FLOW_TASK_UNIFY="""
# Task:
To resolve real-world GitHub issues effectively, the solution can be divided into two primary stages: localization (identifying relevant files and code snippets) and editing (making necessary code changes). 
Your objective is to localize the specific files, classes or functions declarations, and lines of code that need modification or contain key information to resolve the issue.

# Localization Process:
The localization stage can be broken down further into two key parts:
1. Understanding and Reproducing the Problem: 
    - Analyze the Problem Statement to gather a comprehensive list of all files and modules involved.
    - Identify the sequence of execution described in the problem.
2. Identifying Files and Modules for Modification:
    - Pinpoint the specific file(s) and module(s) that require changes to resolve the issue.

Now, given the following GitHub problem description, the focus is on analyzing the problem and reconstructing the flow of execution. 

## Task 1: Understanding and Reproducing the Problem
The task is divided into three steps:
1. Analyze the issue
    - Goal: Fully understand the problem described in the Problem Statement.
    - Action: Carefully read and interpret the problem description. Identify potential causes, such as logical errors, misconfigurations, or faulty dependencies.
2. Extract Keywords and Search for Code References
    - Goal: Collect ALL the relevant files, classes, functions and variable mentioned in the problem.
    - Action: 
        - Extract ALL the file/class/function/variable names appeared the Problem Statement and any other key words (e.g., the function description or the potential name of the module). 
        - Call retrieval-based functions such as `search_in_repo` to gather more information on these components.
3. Walk through the repository by calling tools and Reconstruct the Execution Flow
    - Goal: Understand the flow of execution by identifying the sequence of function calls and class interactions leading to the issue.
    - Action:
        - Identify the main entry point (e.g., class instantiation or method invocation) that triggers the issue.
        - Trace the sequence of events (function calls, function executions, class instantiation, class inheritance) and note how various parts of the code interact.
        - Use tools such as `code_graph_traverse_searcher` to get the full implementation of the modules and to explore files, classes, and methods to fully understand the flow and dependencies within the system.
        - Identify any breakpoints where the issue might arise, based on the reconstructed flow.
Important: Please check carefully that the Execution Flow you generated is related with the problem and don't get bogged down in irrelevant details.

"""


RECONSTRUCT_TAK_OUTPUT_FORMAT="""
## Output Format for Flow of Execution: 

Requirements: 
Use `->` to represent the flow of execution between functions. Each step should show how one function/class interacts with another, using words like `invokes`, `imports`, `inherits from`, or other relevant interactions.
Each function should be listed with its file path, class (if applicable), and function/method name.

Example:
```
full_path1/file1.py(function: MyClass1.entry_function) -> calls -> full_path2/file2.py(function: MyClass2.function_2)
full_path1/file1.py(class: MyClass1) -> inherits from -> full_path3/file3.py(class: MyClass3)
```
Only return the Flow of Execution.
"""


FAKE_USER_MSG_IN_RECONSTRUCT_TAK = (
    'Verify if you have reconstruct the complete execution flow to the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
    'If not, continue searching for additional locations related to the issue.\n'
    'If you think you have solved the task, please send your final answer to user through message and then use the following command to finish: <finish></finish>.\n'
    'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)


SEARCH_AFTER_RECONSTRUCTION_TASK="""
Now, your task is to locate the specific files, functions or lines of code that are relevant to solving the issue.
Continue searching if necessary.

# Output Format for Search Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function name, or line numbers, ordered by importance.

## Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
function: my_function

full_path2/file2.py
line: 76
function: MyClass2.my_method

full_path3/file3.py
line: 24
line: 156
function: my_function
```

Return just the location(s)
"""


FAKE_USER_MSG_FOR_LOC = (
        'Verify if the found locations contain all the necessary information to address the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
        'If not, continue searching for additional locations related to the issue.\n'
        'Verify that you have carefully analyzed the impact of the found locations on the repository, especially their dependencies. '
        'If you think you have solved the task, please send your final answer (including the former answer and reranking) to user through message and then use the following command to finish: <finish></finish>.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)