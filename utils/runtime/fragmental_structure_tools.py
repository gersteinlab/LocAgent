from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)

# 1. UML diagram
    # input: class names
    # output: both; class, function; inherit, contain; -1 
# all functions of class (inherits from ...)
# 2. invoke and reference
# 3. repo structure: downstream; file/dir; contain;
# 4. file structure
# 5. class structure

_GETDIRECTORYSTRUCTURE_DESCRIPTION = """
Displays the directory structure under a specified path in the codebase, including subdirectories and files. The depth of exploration
can be controlled using the `traversal_depth` parameter. If `traversal_depth` is set to -1, the function will recursively explore
all directories and files without any depth limit.

**Usage Example:**
# Retrieve the directory structure of the root directory with a depth limit of 3
get_directory_structure(dir_path='/', traversal_depth=3)

# Retrieve the directory structure of a specific path without any depth limit
get_directory_structure(dir_path='src/', traversal_depth=-1)

**Parameters:**
- `dir_path` (str): The path of the directory to retrieve the structure for. Must be a valid path, formatted as a forward-slash (/) separated string.
- `traversal_depth` (int): Maximum depth of exploration. A value of `-1` indicates unlimited depth, allowing recursive exploration of all subdirectories and files. Must be either `-1` or a non-negative integer (≥ 0). Default is 2.
"""

GetDirectoryStructureTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='get_directory_structure',
        description=_GETDIRECTORYSTRUCTURE_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'dir_path': {
                    'type': 'string',
                    'description': (
                        "The path of the directory to retrieve the structure for. Must be formatted as a forward-slash (/) separated string "
                        "and correspond to a valid directory in the codebase."
                    ),
                },
                'traversal_depth': {
                    'description': (
                        'Maximum depth of traversal. A value of -1 indicates unlimited depth (subject to a maximum limit).'
                        'Must be either `-1` or a non-negative integer (≥ 0).'
                    ),
                    'type': 'integer',
                    'default': 2,
                },
            },
            'required': ['dir_path'],
        },
    ),
)


_GETFILESTRUCTURES_DESCRIPTION = """
Retrieves the skeleton or structure of specified files from the codebase. This includes all classes and functions defined within these files, 
providing a clear overview of their structure without including the full implementation.

**Usage Example:**
# Retrieve the structure of specific files
get_file_structures(file_list=['src/utils/helpers.py', 'src/models/user.py'])

**Parameters:**
- `file_list` (list[str]): A list of file paths for which to retrieve the structure. The array can contain up to 5 file paths.
  Each path must correspond to a valid file within the codebase.
"""

GetFileStructuresTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='get_file_structures',
        description=_GETFILESTRUCTURES_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'file_list': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        "A list of file paths to retrieve the structure of. Each file path should be valid and accessible, "
                        "and the list can contain up to 5 files."
                    ),
                },
            },
            'required': ['file_list'],
        },
    ),
)


_SEARCHCLASSSTRUCTURES_DESCRIPTION = """
Searches the codebase for the skeletons (definitions) of specified classes by their fully qualified names. 
This tool retrieves key details about each class, including file paths, class names, and method signatures, allowing developers to quickly understand class structures.

**Usage Example:**
# Search for the structures of specific classes
search_class_structures(class_names=['src/module1.py:MyClass', 'src/module2.py:AnotherClass'])

# Search for a class structure across the entire codebase
search_class_structures(class_names=['src/module1.py:TargetClass'])

**Parameters:**
- `class_names` (list[str]): A list of fully qualified class names to search for, formatted as "file_path:QualifiedName". 
  For example, "src/module1.py:MyClass" or "src/module2.py:AnotherClass".
"""

SearchClassStructuresTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_class_structures',
        description=_SEARCHCLASSSTRUCTURES_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'class_names': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        "A list of fully qualified class names to search for in the codebase, formatted as 'file_path:QualifiedName'. "
                        "For example, 'src/module1.py:MyClass' or 'src/module2.py:AnotherClass'."
                    ),
                },
            },
            'required': ['class_names'],
        },
    ),
)



_RETRIEVECLASSANCESTORSANDMETHODS_DESCRIPTION = """
Retrieves the ancestor classes and their methods for the specified classes. 
This tool provides a detailed view of the inheritance chain above the specified classes, along with the methods defined within those ancestors.

**Usage Example:**
# Retrieve the ancestors and their methods for specific classes
retrieve_class_ancestors_and_methods(module_names=["src/module3.py:TargetClass"])

**Parameters:**
- `module_names` (list[str]): A list of fully qualified module names to analyze, formatted as "file_path:QualifiedName". 
  For example, ["src/module1.py:MyClass", "src/module2.py:AnotherClass"]. The tool retrieves the ancestor classes and their methods for these classes.
"""

RetrieveClassAncestorsAndMethodsTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='retrieve_class_ancestors_and_methods',
        description=_RETRIEVECLASSANCESTORSANDMETHODS_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'module_names': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'A list of fully qualified module names to analyze, formatted as "file_path:QualifiedName". '
                        'For example, ["src/module1.py:MyClass", "src/module2.py:AnotherClass"]. The tool retrieves the ancestor '
                        'classes and their methods for each class in the list.'
                    ),
                },
            },
            'required': ['module_names'],
        },
    ),
)


_EXPLORECLASSHIERARCHY_DESCRIPTION = """
Explores the class hierarchy, retrieving both ancestor and descendant classes along with their methods. 
This tool provides a full view of the inheritance tree, showing both directions of relationships and the methods within these classes.

**Usage Example:**
# Explore the class hierarchy and methods for specific classes
explore_class_hierarchy(module_names=["src/module3.py:TargetClass"])

**Parameters:**
- `module_names` (list[str]): A list of fully qualified module names to analyze, formatted as "file_path:QualifiedName". 
  For example, ["src/module1.py:MyClass", "src/module2.py:AnotherClass"]. The tool retrieves both ancestor and child classes along with their methods.
"""

ExploreClassHierarchyTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='explore_class_hierarchy',
        description=_EXPLORECLASSHIERARCHY_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'module_names': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'A list of fully qualified module names to analyze, formatted as "file_path:QualifiedName". '
                        'For example, ["src/module1.py:MyClass", "src/module2.py:AnotherClass"]. The tool retrieves both ancestor '
                        'and child classes and their methods for each class in the list.'
                    ),
                },
            },
            'required': ['module_names'],
        },
    ),
)




_SEARCHMODULEREL_DESCRIPTION = """Analyzes the one-hop dependencies of specific classes or functions.
* Identify immediate interactions of the given modules with other parts of the codebase, such as invoked functions, referenced classes, or other dependent components.
* This is useful for quickly understanding direct dependencies and ensuring related components are accounted for in changes or issues.

**Examples:**
search_invoke_and_reference(["path/to/file1.py:function_1", "path/to/file2.py:class_A"])
"""

SearchModuleRelTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='search_invoke_and_reference',
        description=_SEARCHMODULEREL_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'module_names': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'A list of module names to analyze, formatted as "file_path:QualifiedName". '
                        'For example, ["interface/C.py:C.method_a.inner_func", "module/A.py:A.func_b"]. '
                        'Each item defines a module whose one-hop dependencies will be searched.'
                    ),
                },
            },
            'required': ['module_names'],
        },
    ),
)

ALL_FRAGMENTAL_STRUCTURE_TOOLS=[
    'get_directory_structure',
    'get_file_structures',
    'search_class_structures',
    'retrieve_class_ancestors_and_methods',
    'explore_class_hierarchy',
    'search_invoke_and_reference',
]

