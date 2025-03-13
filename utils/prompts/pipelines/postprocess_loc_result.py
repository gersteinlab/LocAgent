SUMMARY_INSTRUCTION_v0 = """
Based on the history messages, please analyze the relevance of each location found to the given GitHub issue, and then generate the Analysis Report about how specific classes, functions, or code blocks contribute to the problem, how they relate to the broader context of the issue, and what impact they may have on resolving it.
The Analysis Report should be ordered by importance.
Every Analysis Report must use this format:
1. The file path where the code resides
2. The specific class name, function name or code blocks being discussed. Avoid vague references to line numbers; instead, quote actual lines of code where needed.
3. Relevance Analysis:
    - Clearly explain how this code or function connects to the issue, why it's important, and how it might be contributing to the problem.
4. Potential Solution Description:
    - Propose one or more potential solutions, addressing how to modify the existing code to solve the issue. 

Here is an example:
{
### Location: 
project/data_parser.py
function: parse_json
lines:
```python
        parsed_data = {}
        for key, value in json_data.items():
            if isinstance(value, dict):
                parsed_data[key] = value  # Directly assigns nested dict
            else:
                parsed_data[key] = process_value(value)
```

### Relevance Analysis: 
    - The parse_json function is responsible for taking raw JSON data and converting it into Python objects for further processing in the application. The GitHub issue reports that certain nested JSON fields are not being parsed correctly, leading to missing or malformed data structures when the JSON contains multiple layers of nesting.

### Potential Solution Description:
    - A potential solution would involve recursively handling nested dictionaries, ensuring that all layers of the JSON are traversed and correctly processed. By modifying the code to recursively call parse_json on nested dictionaries, we can ensure that deeply nested fields are parsed properly.

}

Please be careful of using PROPER INDENTATION while including code lines. If you would like to include the line '        print(x)', you must fully write that out, with all those spaces before the code!
"""

SUMMARY_INSTRUCTION_v0_1 = """
Based on the history messages, please analyze the relevance of each location found to the given GitHub issue, and then generate the Analysis Report about how specific classes, functions, or code blocks contribute to the problem, how they relate to the broader context of the issue, and what impact they may have on resolving it.
The Analysis Report should be structured in order of importance and adhere to the following format:
1. The file path where the code resides.
2. The specific class name, function name being discussed. Avoid using vague references like line numbers. Instead, include the relevant code snippets or logic.
3. Relevance Analysis:
    - Clearly explain how this class or function connects to the issue, and how it might be contributing to the problem.
4. Potential Solution Description:
    - Provide potential ideas to solve the problem.
    - Focus on conceptual solutions-you do not need to write specific code changes.

Here is an example:
{
### Location: 
project/data_parser.py
function: parse_json

### Relevance Analysis: 
    - The parse_json function is responsible for taking raw JSON data and converting it into Python objects for further processing in the application. The GitHub issue reports that certain nested JSON fields are not being parsed correctly, leading to missing or malformed data structures when the JSON contains multiple layers of nesting.

### Potential Solution Description:
    - A potential solution would involve recursively handling nested dictionaries, ensuring that all layers of the JSON are traversed and correctly processed. By modifying the code to recursively call parse_json on nested dictionaries, we can ensure that deeply nested fields are parsed properly.
}
"""

SUMMARY_INSTRUCTION_v1 = """
Based on the history messages, please analyze the relevance of each location found to the given GitHub issue, and then first generate the Analysis Report about how specific classes, functions, or code blocks contribute to the problem, how they relate to the broader context of the issue.
The Analysis Report should be structured in order of importance and adhere to the following format:
1. The file path where the code resides.
2. The specific class name, function name being discussed. Avoid using vague references like line numbers. Instead, include the relevant code snippets or logic.
3. Relevance Analysis:
    - Clearly explain how this class or function connects to the issue, and how it might be contributing to the problem.

Here is an example:
{
## Analysis Report

### Location: 
project/data_parser.py
function: parse_json

### Relevance Analysis: 
    - The parse_json function is responsible for taking raw JSON data and converting it into Python objects for further processing in the application. The GitHub issue reports that certain nested JSON fields are not being parsed correctly, leading to missing or malformed data structures when the JSON contains multiple layers of nesting.
}

After the Analysis Report, generate multiple Potential Solutions, each of which could independently resolve the given issue. 
Every Potential Solutions must use this format:
1. The file path where the code resides.
2. The specific class name, function name being discussed. Avoid using vague references like line numbers. Instead, include the relevant code snippets or logic.
3. Solution Description:
    - Clearly describe the solution, addressing how to modify the existing code to solve the issue. 
    - The description should focus on conceptual approaches that minimize changes to the codebase while still effectively addressing the problem. 
    - You do not need to write specific code changes.

Example:
## Potential Solution 1:
{
### Edit Locations: 
project/data_parser.py
function: parse_json

### Solution Description:
    - A potential solution involves adjusting the `parse_json` function to recursively process nested dictionaries, ensuring all layers of the JSON are correctly handled. 
}

"""

# only analysis
SUMMARY_INSTRUCTION_v2 = """
Based on the history messages, please analyze the relevance of each location found to the given GitHub issue, and then generate the Analysis Report about how specific classes, functions, or code blocks contribute to the problem, how they relate to the broader context of the issue.
The Analysis Report should be structured in order of importance and adhere to the following format:
1. The file path where the code resides.
2. The specific class name, function name being discussed. Avoid using vague references like line numbers. Instead, include the relevant code snippets or logic.
3. Relevance Analysis:
    - Clearly explain how this class or function connects to the issue, and how it might be contributing to the problem.

Here is an example:
{
## Analysis Report

### Location: 
project/data_parser.py
function: parse_json

### Relevance Analysis: 
    - The parse_json function is responsible for taking raw JSON data and converting it into Python objects for further processing in the application. The GitHub issue reports that certain nested JSON fields are not being parsed correctly, leading to missing or malformed data structures when the JSON contains multiple layers of nesting.
}
"""


RERANK_INSTRUCTION="""
Given the GitHub Problem Description, please merge the found locations of identical files, classes, or functions. 
Then, use appropriate tools to analyze the importance of these merged locations and rerank them based on their relevance and importance. 
After reranking, provide a summary listing the locations requiring modification. 
Ensure that no locations listed in the Found locations section are omitted in the output.

### GitHub Problem Description ###
{problem_statement}

### Found locations ###
{found_locations}

### Final Output Format:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function or method name, and line numbers, ordered by importance.

#### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s)
"""