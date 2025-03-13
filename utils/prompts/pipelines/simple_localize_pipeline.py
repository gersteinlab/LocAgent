SEARCH_LOC_TASK_INSTRUCTION="""
# Task:
You will be provided with a GitHub problem description. Your objective is to localize the specific files, classes, functions, or variable declarations that require modification or contain essential information to resolve the issue.

1. Analyze the issue: Understand the problem described in the issue and identify what might be causing it.
2. Extract the Necessary Search Parameters from the issue and call retrieval-based functions:
Determine if specific file types, directories, function or class names or code patterns are mentioned in the issue.
Make sure that at least one of `query`, `code_snippet`, `class_names`, or `function_names` is provided when calling retrival tools like `search_in_repo`.
3. Locate the specific files, functions, methods, or lines of code that are relevant to solving the issue.
"""


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


FAKE_USER_MSG_FOR_LOC = (
        'Verify if the found locations contain all the necessary information to address the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
        'If not, continue searching for additional locations related to the issue.\n'
        'Verify that you have carefully analyzed the impact of the found locations on the repository, especially their dependencies. '
        'If you think you have solved the task, please send your final answer (including the former answer and reranking) to user through message and then use the following command to finish: <finish></finish>.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)