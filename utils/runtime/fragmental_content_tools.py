from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)


_GETFILECONTENT_DESCRIPTION = """
Fetches the entire content of a specified file from the codebase. 
This tool is useful for retrieving the complete file data to understand the context, locate potential issues, or analyze dependencies.

**Usage Example:**
# Retrieve the content of a specific file
get_file_content(file_path='src/utils/helpers.py')

**Parameters:**
- `file_path` (str): The path to the file whose content needs to be retrieved. The file path must be valid and accessible within the codebase.
"""

GetFileContentTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='get_file_content',
        description=_GETFILECONTENT_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': "The path to the file to retrieve the content of. Ensure the file path is valid and accessible.",
                },
            },
            'required': ['file_path'],
        },
    ),
)


_SEARCHCLASS_DESCRIPTION = """
Searches the codebase for the implementation of a specified class by its fully qualified name. 
This tool helps developers locate class definitions, understand their structure, 
and analyze their role within the overall project architecture.

**Usage Example:**
# Search for a class definition by fully qualified name
search_class_content(module_name='src/module1.py:MyClass')

**Parameters:**
- `module_name` (str): The fully qualified name of the class to search for, formatted as "file_path:QualifiedName". 
  For example, "src/module1.py:MyClass". Only one class name should be provided at a time.
"""

SearchClassTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_class_content',
        description=_SEARCHCLASS_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'module_name': {
                    'type': 'string',
                    'description': (
                        'The fully qualified name of the class to search for, formatted as "file_path:QualifiedName". '
                        'For example, "src/module1.py:MyClass".'
                    ),
                },
            },
            'required': ['module_name'],
        },
    ),
)


_SEARCHFUNCTION_DESCRIPTION = """
Searches the codebase for the definitions of specified functions by their fully qualified names and retrieves their implementations 
along with relevant context. This tool allows developers to quickly locate multiple function definitions, understand their behavior, 
and analyze their roles within the codebase.

**Usage Example:**
# Search for functions by their fully qualified names
search_function_contents(module_names=['src/module1.py:ClassName.function_name', 'src/module2.py:global_function'])

**Parameters:**
- `module_names` (list[str]): A list of fully qualified names of functions to search for, formatted as "file_path:QualifiedName". 
  For example, ["src/module1.py:ClassName.function_name", "src/module2.py:global_function"].
"""

SearchFunctionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_function_contents',
        description=_SEARCHFUNCTION_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'module_names': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'A list of fully qualified names of functions to search for, formatted as "file_path:QualifiedName". '
                        'For example, ["src/module1.py:ClassName.function_name", "src/module2.py:global_function"].'
                    ),
                },
            },
            'required': ['module_names'],
        },
    ),
)



_SEARCHKEYWORDSNIPPETS_DESCRIPTION = """
Searches the codebase for specific terms, such as potential class or function names, code snippets, or general keywords, 
and retrieves related code snippets. This tool is useful for locating relevant parts of the codebase based on free-text searches, 
with optional filtering using a file pattern.

**Usage Example:**
# Search for terms across the entire codebase
search_keyword_snippets(search_terms=['config', 'debug'])

# Search for terms in Python files under the 'src' directory
search_snippets_by_keywords(search_terms=['initialize', 'process'], file_pattern='src/**/*.py')

**Parameters:**
- `search_terms` (list[str]): A list of terms to search for. These terms can be potential class or function names, code snippets, or general keywords.
- `file_pattern` (Optional[str]): A glob pattern to filter the search to specific files or directories 
  (e.g., 'src/**/*.py' to search in Python files under the 'src' directory). If not provided, the search includes all files in the codebase.
"""

SearchKeywordSnippetsTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_snippets_by_keywords',
        description=_SEARCHKEYWORDSNIPPETS_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'search_terms': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'A list of terms to search for in the codebase. These terms can include class or function names, '
                        'code snippets, or general keywords.'
                    ),
                },
                'file_pattern': {
                    'type': ['string', 'null'],
                    'description': (
                        'A glob pattern to restrict the search to specific files or directories. '
                        'If not provided, the search includes all files in the codebase.'
                    ),
                },
            },
            'required': ['search_terms'],
        },
    ),
)


_SEARCHCONTEXT_DESCRIPTION = """
Retrieves the surrounding context of specified line numbers within a given file. This tool is useful for understanding the code in context, 
helping developers analyze the logic or dependencies around specific lines of interest.

**Usage Example:**
# Retrieve context for specific line numbers in a file
search_context_by_lines(file_path='src/module1.py', line_nums=[10, 25, 40])

**Parameters:**
- `file_path` (str): The path to the file where the line numbers are located.
- `line_nums` (list[int]): A list of line numbers for which to retrieve context. Multiple line numbers can be queried at once.
"""

SearchLineContextTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_context_by_lines',
        description=_SEARCHCONTEXT_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': (
                        'The path to the file where the line numbers are located. This must be a valid file in the codebase.'
                    ),
                },
                'line_nums': {
                    'type': 'array',
                    'items': {'type': 'integer'},
                    'description': (
                        'A list of line numbers to retrieve context for. The context typically includes a few lines '
                        'before and after each specified line number.'
                    ),
                },
            },
            'required': ['file_path', 'line_nums'],
        },
    ),
)


ALL_FRAGMENTAL_CONTENT_TOOLS = [
    'get_file_content',
    'search_class_content',
    'search_function_contents',
    'search_snippets_by_keywords',
    'search_context_by_lines'
]

# search tool
# 6. file content
# 7. class content
# 8. function content
# 9. search keywords
# 10. search line number within a file