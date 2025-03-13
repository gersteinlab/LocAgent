FULL_QUALIFIED_NAME_INFER_AFTER_EXTRACTION_TASK="""
To to fully understand the issue, based on the extracted keywords and line numbers, search for code references to identify each referenced module's specific location.
Pay special attention to distinguishing between modules that share the same names, using the context and execution flow described in the problem statement to accurately determine the specific module being referred to.

## Steps to Follow:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Search for code references to locate each module, class, function, and variable, ensuring you distinguish between similarly named elements based on the context.
3. Analyze the execution flow or sequence of operations described in the problem to gain insights into which specific module or component is being referenced.

## Output Format
Represent each module by combining its file path and its Qualified Name (QN) relative to the file.
For example, for a function `calculate_sum` inside a class `MathUtils` located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py:MathUtils.calculate_sum'.
As the output, list ALL the relevant files, modeuls(classes, functions and variable), wrapped with triple backticks ```

### Examples:
```
full_path1/file1.py
full_path1/file2.py:MyClass1.my_function1
full_path2/file3.py:my_function2
full_path2/file4.py:variable1
```
"""


FULL_QUALIFIED_NAME_INFER_TASK="""
Given the following GitHub problem description, to fully understand the issue, please analyze the problem statement and identify each referenced module's location. 
Pay special attention to distinguishing between modules that share the same names, using the context and execution flow described in the problem statement to accurately determine the specific module being referred to.

## Steps to Follow:
1. Classify the problem statement into the following categories: problem description, error trace, code to reproduce the bug, and additional context.
    Then identify a list of modules in the "{package_name}" package mentioned by each category.
    {output_format}
2. Search for code references to locate each module, class, function, and variable, ensuring you distinguish between similarly named elements based on the context.
3. Analyze the execution flow or sequence of operations described in the problem to gain insights into which specific module or component is being referenced.

## Output Format
Represent each module by combining its file path and its Qualified Name (QN) relative to the file.
For example, for a function `calculate_sum` inside a class `MathUtils` located in `src/helpers/math_helpers.py`, represent it as: 'src/helpers/math_helpers.py:MathUtils.calculate_sum'.
As the output, list ALL the relevant files, modeuls(classes, functions and variable), wrapped with triple backticks ```

### Examples:
```
full_path1/file1.py
full_path1/file2.py:MyClass1.my_function1
full_path2/file3.py:my_function2
full_path2/file4.py:variable1
```
"""


REWRITE_PR_TASK="Given the following GitHub problem description, classify the problem statement into the following categories: "\
                "Problem description, error trace, code to reproduce the bug, and additional context."


KEYWORD_EXTRACTION_TASK="Then identify a list of modules in the '{package_name}' package mentioned by each category.\n"
OUTPUT_FORMAT_KEYWORD_EXTRACTION=r"""
Output in the following format: 
[{"keyword": "", "possible_file_path": "", "possible_line_numbers": []}]
"""


SEARCH_INSTRUCTION="""
Now, your task is to locate specific files, functions, or lines of code that either require modification or contain critical information for resolving the issue.
-   The bug could originate within the modules(including inner implementations, such as method logic or data handling) mentioned in the problem, 
    or it may arise from upstream dependencies (code or data that flows into the modules) or downstream effects (code or behavior affected by the modules).
    Consider viewing the upstream and downstream dependencies of key modules.
-   Additionally, be aware that some problem statements may propose or necessitate the addition of fields, functions, or variables to support the desired functionality. 
    In such cases, identify where these changes should be introduced and how they impact the existing codebase.

Continue searching if necessary.
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


FAKE_USER_MSG_FOR_MLOC = (
    'Ensure you have identified all relevant modules referenced in the problem, particularly distinguishing between modules with identical names. '
    'If not, analyze the context thoroughly and the described execution flow to accurately pinpoint the specific module being referenced.\n'
    'If you think you have solved the task, please send your final answer to the user through message and then use the following command to finish: <finish></finish>.\n'
    'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)


FAKE_USER_MSG_FOR_LOC = (
        'Verify if the found locations contain all the necessary information to address the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
        'If not, continue searching for additional locations related to the issue.\n'
        'Verify that you have carefully analyzed the impact of the found locations on the repository, especially their dependencies. '
        'If you think you have solved the task, please send your final answer (including the former answer and reranking) to user through message and then use the following command to finish: <finish></finish>.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)