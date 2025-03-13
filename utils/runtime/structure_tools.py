from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)

_TREE_EXAMPLE_simple = """
Example Usage:
1. Exploring Downstream Dependencies:
    ```
    explore_tree_structure(
        start_entities=['src/module_a.py:ClassA'],
        direction='downstream',
        traversal_depth=2,
        dependency_type_filter=['invokes', 'imports']
    )
    ```
2. Exploring the repository structure from the root directory (/) up to two levels deep:
    ```
    explore_tree_structure(
      start_entities=['/'],
      traversal_depth=2,
      dependency_type_filter=['contains']
    )
    ```
3. Generate Class Diagrams:
    ```
    explore_tree_structure(
        start_entities=selected_entity_ids,
        direction='both',
        traverse_depth=-1,
        dependency_type_filter=['inherits']
    )
    ```
"""

_STRUCTURE_EXPLORER_DESCRIPTION_simple = """
A unified tool that traverses a pre-built code graph to retrieve dependency structure around specified entities, 
with options to explore upstream or downstream, and control traversal depth and filters for entity and dependency types.
"""

_GRAPH_STRUCTURE_EXPLORER_DESCRIPTION_DEPRECATED = """
Unified repository searching tool that traverses a pre-built code graph to retrieve code contents and their interactions.
This function performs a BFS traversal on a code graph starting from specified nodes, allowing control over traversal direction, depth (hops), and filtering by node and edge types. 
It returns node data list and edge list representing the traversed subgraph.

Code Graph Definition:
* Node Types: 'directory', 'file', 'class', 'function'.
* Edge Types: 'contains', 'imports', 'invokes', 'inherits'.
* Hierarchy:
    - Directories contain files and subdirectories.
    - Files contain classes and functions.
    - Classes contain inner classes and methods.
    - Functions can contain inner functions.
* Interactions:
    - Files can import classes and functions.
    - Classes can inherit from other classes.
    - Classes and functions can invoke others (invocations in a class's `__init__` are attributed to the class).

Node ID:
* Unique identifier including file path and module path.
* Here's an example of a Node ID: `"interface/C.py:C.method_a.inner_func"` identifies function `inner_func` within `method_a` of class `C` in `"interface/C.py"`.
* Always use complete Node IDs in `input_node_ids` if possible.

Usage Examples:
* Get Repository Structure:
    ```
    code_graph_traverse_searcher(
      input_node_ids=['/'],
      traverse_hops=-1,
      node_type_filter=('directory', 'file'),
      edge_type_filter=('contains',)
    )
    ```
* Retrieve Code Content:
    ```
    code_graph_traverse_searcher(
      input_node_ids=selected_node_ids,
      traverse_hops=0,
      return_code_content=True
    )
    ```
* Search Invocations:
    ```
    code_graph_traverse_searcher(
        input_node_ids=selected_node_ids,
        traverse_hops=call_trace_length,
        edge_type_filter=('invokes',)
    )
    ```
    Add 'contains' to edge_type_filter for more class invocations.
* Search References:
    ```
    code_graph_traverse_searcher(
        input_node_ids=selected_node_ids,
        direction='backward',
        traverse_hops=call_trace_length,
        edge_type_filter=('invokes',)
    )
    ```
* Generate Class Diagrams:
    ```
    code_graph_traverse_searcher(
        input_node_ids=selected_node_ids,
        direction='bidirectional',
        traverse_hops=-1,
        edge_type_filter=('inherits',)
    )
    ```

Notes:
* State Persistence: Maintains state across calls and user interactions.
* Traversal Constraints: Only include nodes and edges whose type is in `node_type_filter` and `edge_type_filter`, respectively. Include all types if the filter is None. 
* Managing Output Size: To handle large outputs, first identify nodes through traversal, then request code content using return_code_content=True.
* Possible Noise: The output can contain more nodes/edges than actual. Remember to select the correct nodes/edges if necessary.
"""


_STRUCTURE_EXPLORER_DESCRIPTION = """
Unified repository exploring tool that traverses a pre-built code graph to retrieve dependency structure around specified entities.
The search can be controlled to traverse upstream (exploring dependencies that entities rely on) or downstream (exploring how entities impact others), with optional limits on traversal depth and filters for entity and dependency types.

Code Graph Definition:
* Entity Types: 'directory', 'file', 'class', 'function'.
* Dependency Types: 'contains', 'imports', 'invokes', 'inherits'.
* Hierarchy:
    - Directories contain files and subdirectories.
    - Files contain classes and functions.
    - Classes contain inner classes and methods.
    - Functions can contain inner functions.
* Interactions:
    - Files/classes/functions can import classes and functions.
    - Classes can inherit from other classes.
    - Classes and functions can invoke others (invocations in a class's `__init__` are attributed to the class).
Entity ID:
* Unique identifier including file path and module path.
* Here's an example of an Entity ID: `"interface/C.py:C.method_a.inner_func"` identifies function `inner_func` within `method_a` of class `C` in `"interface/C.py"`.

Notes:
* Traversal Control: The `traversal_depth` parameter specifies how deep the function should explore the graph starting from the input entities.
* Filtering: Use `entity_type_filter` and `dependency_type_filter` to narrow down the scope of the search, focusing on specific entity types and relationships.

"""

_GRAPH_EXAMPLE = """
Example Usage:
1. Exploring Outward Dependencies:
    ```
    explore_graph_structure(
        start_entities=['src/module_a.py:ClassA'],
        direction='downstream',
        traversal_depth=2,
        dependency_type_filter=['invokes', 'imports']
    )
    ```
    This retrieves the dependencies of `ClassA` up to 2 levels deep, focusing only on classes and functions with 'invokes' and 'imports' relationships.

2. Exploring Inward Dependencies:
    ```
    explore_graph_structure(
        start_entities=['src/module_b.py:FunctionY'],
        direction='upstream',
        traversal_depth=-1
    )
    ```
    This finds all entities that depend on `FunctionY` without restricting the traversal depth.
3. Exploring Repository Structure:
    ```
    explore_graph_structure(
      start_entities=['/'],
      traversal_depth=2,
      dependency_type_filter=['contains']
    )
    ```
    This retrieves the tree repository structure from the root directory (/), traversing up to two levels deep and focusing only on 'contains' relationship.
4. Generate Class Diagrams:
    ```
    explore_graph_structure(
        start_entities=selected_entity_ids,
        direction='both',
        traverse_depth=-1,
        dependency_type_filter=['inherits']
    )
    ```
    This finds class diagrams by exploring both downstream and upstream 'inherits' dependencies for selected entities with no depth restriction (traverse_depth=-1).
"""

_TREE_EXAMPLE = """
Example Usage:
1. Exploring Outward Dependencies:
    ```
    explore_tree_structure(
        start_entities=['src/module_a.py:ClassA'],
        direction='downstream',
        traversal_depth=2,
        dependency_type_filter=['invokes', 'imports']
    )
    ```
    This retrieves the dependencies of `ClassA` up to 2 levels deep, focusing only on classes and functions with 'invokes' and 'imports' relationships.

2. Exploring Inward Dependencies:
    ```
    explore_tree_structure(
        start_entities=['src/module_b.py:FunctionY'],
        direction='upstream',
        traversal_depth=-1
    )
    ```
    This finds all entities that depend on `FunctionY` without restricting the traversal depth.
3. Exploring Repository Structure:
    ```
    explore_tree_structure(
      start_entities=['/'],
      traversal_depth=2,
      dependency_type_filter=['contains']
    )
    ```
    This retrieves the tree repository structure from the root directory (/), traversing up to two levels deep and focusing only on 'contains' relationship.
4. Generate Class Diagrams:
    ```
    explore_tree_structure(
        start_entities=selected_entity_ids,
        direction='both',
        traverse_depth=-1,
        dependency_type_filter=['inherits']
    )
    ```
"""

_STRUCTURE_EXPLORER_PARAMETERS = {
    'type': 'object',
    'properties': {
        'start_entities': {
            'description': (
                'List of entities (e.g., class, function, file, or directory paths) to begin the search from.\n'
                'Entities representing classes or functions must be formatted as "file_path:QualifiedName" (e.g., `interface/C.py:C.method_a.inner_func`).\n'
                'For files or directories, provide only the file or directory path (e.g., `src/module_a.py` or `src/`).'
            ),
            'type': 'array',
            'items': {'type': 'string'},
        },
        'direction': {
            'description': (
                'Direction of traversal in the code graph; allowed options are: `upstream`, `downstream`, `both`.\n'
                "- 'upstream': Traversal to explore dependencies that the specified entities rely on (how they depend on others).\n"
                "- 'downstream': Traversal to explore the effects or interactions of the specified entities on others (how others depend on them).\n"
                "- 'both': Traversal on both direction."
            ),
            'type': 'string',
            'enum': ['upstream', 'downstream', 'both'],
            'default': 'downstream',
        },
        'traversal_depth': {
            'description': (
                'Maximum depth of traversal. A value of -1 indicates unlimited depth (subject to a maximum limit).'
                'Must be either `-1` or a non-negative integer (≥ 0).'
            ),
            'type': 'integer',
            'default': 2,
        },
        'entity_type_filter': {
            'description': (
                "List of entity types (e.g., 'class', 'function', 'file', 'directory') to include in the traversal. If None, all entity types are included."
            ),
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'default': None,
        },
        'dependency_type_filter': {
            'description': (
                "List of dependency types (e.g., 'contains', 'imports', 'invokes', 'inherits') to include in the traversal. If None, all dependency types are included."
            ),
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'default': None,
        },
    },
    'required': ['start_entities'],
}

ExploreGraphStructure = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='explore_graph_structure',
        description=_STRUCTURE_EXPLORER_DESCRIPTION + _GRAPH_EXAMPLE,
        parameters=_STRUCTURE_EXPLORER_PARAMETERS
    ),
)

_REPO_STRUCTURE_EXPLORER_DESCRIPTION = """This function explores the structure of a code repository starting from a specified node, which can be a directory, file, or class name.

The function builds a tree-like structure to represent the relationships and hierarchy within the repository, excluding test files. 
It returns a structured string showing the repository's organization from the specified start entity.
"""

RepoStructureExplorerTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='explore_repo_structure',
        description=_REPO_STRUCTURE_EXPLORER_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'start_entity': {
                    'type': 'string',
                    'description': (
                        'Entity to start exploration, such as a directory, file, or specific class name. Default is "/".\n'
                        'Entity representing classes or functions must be formatted as "file_path:QualifiedName" (e.g., `interface/C.py:C.method_a.inner_func`).\n'
                        'For file or directory, provide only the file or directory path (e.g., `src/module_a.py` or `src/`).'
                    ),
                },
                'traversal_depth': {
                    'type': 'integer',
                    'description': 'Maximum depth of exploration from the start node. Use -1 for unlimited depth. Must be either `-1` or a non-negative integer (≥ 0).',
                    'default': 3,
                },
            },
            'required': ['start_entity'],
        },
    ),
)


ExploreTreeStructure = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='explore_tree_structure',
        description=_STRUCTURE_EXPLORER_DESCRIPTION + _TREE_EXAMPLE,
        parameters=_STRUCTURE_EXPLORER_PARAMETERS
    ),
)


ExploreTreeStructure_simple = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='explore_tree_structure',
        description=_STRUCTURE_EXPLORER_DESCRIPTION_simple + _TREE_EXAMPLE_simple,
        parameters=_STRUCTURE_EXPLORER_PARAMETERS
    ),
)


_STRUCTURE_EXPLORER_SPECIFIC_HOP_DESCRIPTION = """
Unified repository exploration tool for {hop_num}-hop dependency traversal within a pre-built code graph. 
This tool identifies direct relationships between specified entities and their immediate dependencies or dependents.
The search can be controlled to traverse upstream (exploring dependencies that entities rely on) or downstream (exploring how entities impact others), with optional filters for entity and dependency types.

Code Graph Definition:
* Entity Types: 'directory', 'file', 'class', 'function'.
* Dependency Types: 'contains', 'imports', 'invokes', 'inherits'.
* Hierarchy:
    - Directories contain files and subdirectories.
    - Files contain classes and functions.
    - Classes contain inner classes and methods.
    - Functions can contain inner functions.
* Interactions:
    - Files/classes/functions can import classes and functions.
    - Classes can inherit from other classes.
    - Classes and functions can invoke others (invocations in a class's `__init__` are attributed to the class).
Entity ID:
* Unique identifier including file path and module path.
* Here's an example of an Entity ID: `"interface/C.py:C.method_a.inner_func"` identifies function `inner_func` within `method_a` of class `C` in `"interface/C.py"`.

Notes:
* Filtering: Use `entity_type_filter` and `dependency_type_filter` to narrow down the scope of the search, focusing on specific entity types and relationships.
"""


_TREE_SPECIFIC_HOP_EXAMPLE = """
Example Usage:
1. Exploring Outward Dependencies:
    ```
    {func_name}(
        start_entities=['src/module_a.py:ClassA'],
        direction='downstream',
        dependency_type_filter=['invokes', 'imports']
    )
    ```
    This retrieves the dependencies of `ClassA` up to {hop_num} levels deep, focusing only on classes and functions with 'invokes' and 'imports' relationships.

2. Exploring Repository Structure:
    ```
    {func_name}(
      start_entities=['/'],
      dependency_type_filter=['contains']
    )
    ```
    This retrieves the tree repository structure from the root directory (/), traversing up to {hop_num} levels deep and focusing only on 'contains' relationship.

4. Generate Class Diagrams:
    ```
    {func_name}(
        start_entities=selected_entity_ids,
        direction='both',
        dependency_type_filter=['inherits']
    )
    ```
    This finds class diagrams by exploring both inward and outward 'inherits' dependencies for selected entities with up to {hop_num} levels deep.
"""

_STRUCTURE_EXPLORER_SPECIFIC_HOP_PARAMETERS = {
    'type': 'object',
    'properties': {
        'start_entities': {
            'description': (
                'List of entities (e.g., class, function, file, or directory paths) to begin the search from.\n'
                'Entities representing classes or functions must be formatted as "file_path:QualifiedName" (e.g., `interface/C.py:C.method_a.inner_func`).\n'
                'For files or directories, provide only the file or directory path (e.g., `src/module_a.py` or `src/`).'
            ),
            'type': 'array',
            'items': {'type': 'string'},
        },
        'direction': {
            'description': (
                'Direction of traversal in the code graph; allowed options are: `upstream`, `downstream`, `both`.\n'
                "- 'upstream': Traversal to explore dependencies that the specified entities rely on (how they depend on others).\n"
                "- 'downstream': Traversal to explore the effects or interactions of the specified entities on others (how others depend on them).\n"
                "- 'both': Traversal on both direction."
            ),
            'type': 'string',
            'enum': ['upstream', 'downstream', 'both'],
            'default': 'downstream',
        },
        # 'traversal_depth': {
        #     'description': (
        #         'Maximum depth of traversal. A value of -1 indicates unlimited depth (subject to a maximum limit).'
        #         'Must be either `-1` or a non-negative integer (≥ 0).'
        #     ),
        #     'type': 'integer',
        #     'default': 2,
        # },
        'entity_type_filter': {
            'description': (
                "List of entity types (e.g., 'class', 'function', 'file', 'directory') to include in the traversal. If None, all entity types are included."
            ),
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'default': None,
        },
        'dependency_type_filter': {
            'description': (
                "List of dependency types (e.g., 'contains', 'imports', 'invokes', 'inherits') to include in the traversal. If None, all dependency types are included."
            ),
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'default': None,
        },
    },
    'required': ['start_entities'],
}

ExploreOneHopStructure = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='explore_one_hop_structure',
        description=(_STRUCTURE_EXPLORER_SPECIFIC_HOP_DESCRIPTION.format(hop_num = 'one') + \
                     _TREE_SPECIFIC_HOP_EXAMPLE.format(
                                                    func_name='explore_one_hop_structure',
                                                    hop_num='one',
                                                    )
        ),
        parameters=_STRUCTURE_EXPLORER_SPECIFIC_HOP_PARAMETERS
    ),
)

ExploreTwoHopStructure = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='explore_two_hop_structure',
        description=(_STRUCTURE_EXPLORER_SPECIFIC_HOP_DESCRIPTION.format(hop_num = 'two') + \
                     _TREE_SPECIFIC_HOP_EXAMPLE.format(
                                                    func_name='explore_two_hop_structure',
                                                    hop_num = 'two',
                                                    )
        ),
        parameters=_STRUCTURE_EXPLORER_SPECIFIC_HOP_PARAMETERS
    ),
)

ExploreThreeHopStructure = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='explore_three_hop_structure',
        description=(_STRUCTURE_EXPLORER_SPECIFIC_HOP_DESCRIPTION.format(hop_num = 'three') + \
                     _TREE_SPECIFIC_HOP_EXAMPLE.format(
                                                    func_name='explore_three_hop_structure',
                                                    hop_num = 'three',
                                                    )
        ),
        parameters=_STRUCTURE_EXPLORER_SPECIFIC_HOP_PARAMETERS
    ),
)


SPECIFIC_HOP_STRUCTURE_EXPLORERS = [
    'explore_one_hop_structure', 
    'explore_two_hop_structure', 
    'explore_three_hop_structure' ]