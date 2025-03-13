# AGENT_CLS_TO_INST_SUFFIX = "When you think you have done the task, please use the following command to finish: <finish></finish>.\nLet's think step by step!\n"
# AGENT_CLS_TO_INST_SUFFIX = "Let's think step by step!\n"
# PR_TEMPLATE = """
# --- BEGIN PROBLEM STATEMENT ---
# Title: {title}

# {description}
# --- END PROBLEM STATEMENT ---

# """

# SYSTEM_PROMPT="""You're an experienced software tester and static analysis expert. 
# Given the problem offered by the user, please perform a thorough static analysis and to localize the bug in this repository using the available tools.
# Analyze the execution flow of this code step by step, as if you were a human tester mentally running it.

# Focus on:
# - Tracing the flow of execution through critical paths, conditions, loops, and function calls.
# - Identifying any deviations, potential errors, or unexpected behavior that could contribute to the issue.
# - Considering how dynamic binding, late resolution, or other runtime behavior may influence the code's behavior.
# - Highlighting possible root causes or key areas for further inspection.
# """

# OUTPUT_FORMAT="""
# ## Output Format for Flow of Execution: 

# Requirements: 
# Use `->` to represent the flow of execution between functions. Each step should show how one function/class interacts with another, using words like `invokes`, `imports`, `inherits from`, or other relevant interactions.
# Each function should be listed with its file path, class (if applicable), and function/method name.

# Example:
# ```
# full_path1/file1.py(function: MyClass1.entry_function) -> calls -> full_path2/file2.py(function: MyClass2.function_2)
# full_path1/file1.py(class: MyClass1) -> inherits from -> full_path3/file3.py(class: MyClass3)
# ```
# Only return the Flow of Execution.
# """

OUTPUT_FORMAT_LOC="""
# Output Format for Search Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function name, or line numbers, ordered by importance.

## Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
function: my_function1

full_path2/file2.py
line: 76
function: MyClass2.my_function2

full_path3/file3.py
line: 24
line: 156
function: my_function3
```

Return just the location(s)
"""

# SEARCH_LOC_TASK_INSTRUCTION="""
# # Task:
# You will be provided with a GitHub problem description. Your objective is to localize the specific files, classes, functions, or variable declarations that require modification or contain essential information to resolve the issue.

# 1. Analyze the issue: Understand the problem described in the issue and identify what might be causing it.
# 2. Extract the Necessary Search Parameters from the issue and call retrieval-based functions:
# Determine if specific file types, directories, function or class names or code patterns are mentioned in the issue.
# Make sure that at least one of `query`, `code_snippet`, `class_names`, or `function_names` is provided when calling retrival tools like `search_in_repo`.
# 3. Locate the specific files, functions, methods, or lines of code that are relevant to solving the issue.
# """

SEARCH_INSTRUCTION="""
Now, your task is to locate specific files, functions, or lines of code that either require modification or contain critical information for resolving the issue.
-   The bug could originate within the modules(including inner implementations, such as method logic or data handling) mentioned in the problem, 
    or it may arise from upstream dependencies (code or data that flows into the modules) or downstream effects (code or behavior affected by the modules).
    Consider viewing the upstream and downstream dependencies of key modules.
-   Additionally, be aware that some problem statements may propose or necessitate the addition of fields, functions, or variables to support the desired functionality. 
    In such cases, identify where these changes should be introduced and how they impact the existing codebase.

Continue searching if necessary.
"""

SEARCH_INSTRUCTION_AFTER_RECONSTRUCTION="""
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


# FAKE_USER_MSG = (
#         'Verify if you have reconstruct the complete execution flow to the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
#         'If not, continue searching for additional locations related to the issue.\n'
#         'If you think you have solved the task, please send your final answer to user through message and then use the following command to finish: <finish></finish>.\n'
#         'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
#     )


FAKE_USER_MSG_FOR_MLOC = (
    'Ensure you have identified all relevant modules referenced in the problem, particularly distinguishing between modules with identical names. '
    'If not, analyze the context thoroughly and the described execution flow to accurately pinpoint the specific module being referenced.\n'
    'If you think you have solved the task, please send your final answer to the user through message and then use the following command to finish: <finish></finish>.\n'
    'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)


# FAKE_USER_MSG_FOR_LOC = (
#         'Verify if the found locations contain all the necessary information to address the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
#         'If not, continue searching for additional locations related to the issue.\n'
#         'Verify that you have carefully analyzed the impact of the found locations on the repository, especially their dependencies. '
#         'If you think you have solved the task, please send your final answer (including the former answer and reranking) to user through message and then use the following command to finish: <finish></finish>.\n'
#         'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
# )

# RECONSTRUCT_FLOW_TASK="""
# # Task:
# To resolve real-world GitHub issues effectively, the solution can be divided into two primary stages: localization (identifying relevant files and code snippets) and editing (making necessary code changes). 
# Your objective is to localize the specific files, classes or functions declarations, and lines of code that need modification or contain key information to resolve the issue.

# # Localization Process:
# The localization stage can be broken down further into two key parts:
# 1. Understanding and Reproducing the Problem: 
#     - Analyze the Problem Statement to gather a comprehensive list of all files and modules involved.
#     - Identify the sequence of execution described in the problem.
# 2. Identifying Files and Modules for Modification:
#     - Pinpoint the specific file(s) and module(s) that require changes to resolve the issue.

# Now, given the following GitHub problem description, the focus is on analyzing the problem and reconstructing the flow of execution. 

# ## Task 1: Understanding and Reproducing the Problem
# The task is divided into three steps:
# 1. Analyze the issue
#     - Goal: Fully understand the problem described in the Problem Statement.
#     - Action: Carefully read and interpret the problem description. Identify potential causes, such as logical errors, misconfigurations, or faulty dependencies.
# 2. Extract Keywords and Search for Code References
#     - Goal: Collect ALL the relevant files, classes, functions and variable mentioned in the problem.
#     - Action: 
#         - Extract ALL the file/class/function/variable names appeared the Problem Statement and any other key words (e.g., the function description or the potential name of the module). 
#         - Call retrieval-based functions such as `search_in_repo`, `get_file_structures` or `search_class_structures` to gather more information on these components.
# 3. Walk through the repository by calling tools and Reconstruct the Execution Flow
#     - Goal: Understand the flow of execution by identifying the sequence of function calls and class interactions leading to the issue.
#     - Action:
#         - Identify the main entry point (e.g., class instantiation or method invocation) that triggers the issue.
#         - Trace the sequence of events (function calls, function executions, class instantiation, class inheritance) and note how various parts of the code interact.
#         - Use tools such as `get_file_content`, `search_class` or `search_method` to get the full implementation of the modules and to explore files, classes, and methods to fully understand the flow and dependencies within the system.
#         - Identify any breakpoints where the issue might arise, based on the reconstructed flow.
# Important: Please check carefully that the Execution Flow you generated is related with the problem and don't get bogged down in irrelevant details.

# """


# RECONSTRUCT_FLOW_TASK_GENERAL="""
# # Task:
# To resolve real-world GitHub issues effectively, the solution can be divided into two primary stages: localization (identifying relevant files and code snippets) and editing (making necessary code changes). 
# Your objective is to localize the specific files, classes or functions declarations, and lines of code that need modification or contain key information to resolve the issue.

# # Localization Process:
# The localization stage can be broken down further into two key parts:
# 1. Understanding and Reproducing the Problem: 
#     - Analyze the Problem Statement to identify all relevant files and modules.
#     - Trace the described sequence of execution to understand how it leads to the issue.
# 2. Identifying Files and Modules for Modification:
#     - Pinpoint the specific file(s) and module(s) that require changes to resolve the issue.

# ## Task 1: Understanding and Reproducing the Problem
# Now, given the following GitHub problem description, the focus is on analyzing the problem and reconstructing the flow of execution. 
# 1. Analyze the Issue:
#     - Thoroughly review the problem statement to understand the issue.
# 2. Extract Keywords and Search for Code References:
#     - Collect ALL the relevant files, classes, functions and variable mentioned in the Problem Statement to gather more context of the problem.
# 3. Walk through the repository and Reconstruct the Execution Flow
#     - Goal: Understand the flow of execution by identifying the sequence of function calls and class interactions leading to the issue.
#     - Action:
#         - Identify the main entry point (e.g., class instantiation or method invocation) that triggers the issue.
#         - Trace the sequence of events (function calls, class instantiation, class inheritance, etc.) to see how various parts of the code interact.
#         - Identify potential breakpoints causing the issue based on the reconstructed flow.
# Important: Ensure the reconstructed execution flow is directly related to the problem, avoiding irrelevant details.

# """


# RECONSTRUCT_FLOW_TASK_UNIFY="""
# # Task:
# To resolve real-world GitHub issues effectively, the solution can be divided into two primary stages: localization (identifying relevant files and code snippets) and editing (making necessary code changes). 
# Your objective is to localize the specific files, classes or functions declarations, and lines of code that need modification or contain key information to resolve the issue.

# # Localization Process:
# The localization stage can be broken down further into two key parts:
# 1. Understanding and Reproducing the Problem: 
#     - Analyze the Problem Statement to gather a comprehensive list of all files and modules involved.
#     - Identify the sequence of execution described in the problem.
# 2. Identifying Files and Modules for Modification:
#     - Pinpoint the specific file(s) and module(s) that require changes to resolve the issue.

# Now, given the following GitHub problem description, the focus is on analyzing the problem and reconstructing the flow of execution. 

# ## Task 1: Understanding and Reproducing the Problem
# The task is divided into three steps:
# 1. Analyze the issue
#     - Goal: Fully understand the problem described in the Problem Statement.
#     - Action: Carefully read and interpret the problem description. Identify potential causes, such as logical errors, misconfigurations, or faulty dependencies.
# 2. Extract Keywords and Search for Code References
#     - Goal: Collect ALL the relevant files, classes, functions and variable mentioned in the problem.
#     - Action: 
#         - Extract ALL the file/class/function/variable names appeared the Problem Statement and any other key words (e.g., the function description or the potential name of the module). 
#         - Call retrieval-based functions such as `search_in_repo` to gather more information on these components.
# 3. Walk through the repository by calling tools and Reconstruct the Execution Flow
#     - Goal: Understand the flow of execution by identifying the sequence of function calls and class interactions leading to the issue.
#     - Action:
#         - Identify the main entry point (e.g., class instantiation or method invocation) that triggers the issue.
#         - Trace the sequence of events (function calls, function executions, class instantiation, class inheritance) and note how various parts of the code interact.
#         - Use tools such as `code_graph_traverse_searcher` to get the full implementation of the modules and to explore files, classes, and methods to fully understand the flow and dependencies within the system.
#         - Identify any breakpoints where the issue might arise, based on the reconstructed flow.
# Important: Please check carefully that the Execution Flow you generated is related with the problem and don't get bogged down in irrelevant details.

# """


# FULL_QUALIFIED_NAME_INFER_AFTER_EXTRACTION_TASK="""
# To to fully understand the issue, based on the extracted keywords and line numbers, search for code references to identify each referenced module's specific location.
# Pay special attention to distinguishing between modules that share the same names, using the context and execution flow described in the problem statement to accurately determine the specific module being referred to.

# ## Steps to Follow:
# 1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
# 2. Search for code references to locate each module, class, function, and variable, ensuring you distinguish between similarly named elements based on the context.
# 3. Analyze the execution flow or sequence of operations described in the problem to gain insights into which specific module or component is being referenced.

# ## Output Format
# Represent each module by combining its file path and its Qualified Name (QN) relative to the file.
# For example, for a function `calculate_sum` inside a class `MathUtils` located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py:MathUtils.calculate_sum'.
# As the output, list ALL the relevant files, modeuls(classes, functions and variable), wrapped with triple backticks ```

# ### Examples:
# ```
# full_path1/file1.py
# full_path1/file2.py:MyClass1.my_function1
# full_path2/file3.py:my_function2
# full_path2/file4.py:variable1
# ```
# """


# FULL_QUALIFIED_NAME_INFER_TASK="""
# Given the following GitHub problem description, to fully understand the issue, please analyze the problem statement and identify each referenced module's location. 
# Pay special attention to distinguishing between modules that share the same names, using the context and execution flow described in the problem statement to accurately determine the specific module being referred to.

# ## Steps to Follow:
# 1. Classify the problem statement into the following categories: problem description, error trace, code to reproduce the bug, and additional context.
#     Then identify a list of modules in the "{package_name}" package mentioned by each category.
#     {output_format}
# 2. Search for code references to locate each module, class, function, and variable, ensuring you distinguish between similarly named elements based on the context.
# 3. Analyze the execution flow or sequence of operations described in the problem to gain insights into which specific module or component is being referenced.

# ## Output Format
# Represent each module by combining its file path and its Qualified Name (QN) relative to the file.
# For example, for a function `calculate_sum` inside a class `MathUtils` located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py:MathUtils.calculate_sum'.
# As the output, list ALL the relevant files, modeuls(classes, functions and variable), wrapped with triple backticks ```

# ### Examples:
# ```
# full_path1/file1.py
# full_path1/file2.py:MyClass1.my_function1
# full_path2/file3.py:my_function2
# full_path2/file4.py:variable1
# ```
# """


# REWRITE_PR_TASK="Given the following GitHub problem description, classify the problem statement into the following categories: "\
#                 "Problem description, error trace, code to reproduce the bug, and additional context."


# KEYWORD_EXTRACTION_TASK="Then identify a list of modules in the '{package_name}' package mentioned by each category."
# OUTPUT_FORMAT_KEYWORD_EXTRACTION=r"""
# Output in the following format: 
# [{"keyword": "", "possible_file_path": "", "possible_line_numbers": []}]
# """