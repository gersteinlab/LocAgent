
"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ModelResponse,
)
import logging

from utils.actions.action import (
    Action,
    FinishAction,
    # AgentDelegateAction,
    # CmdRunAction,
    # FileEditAction,
    IPythonRunCellAction,
    MessageAction,
)
from utils.runtime.structure_tools import (
    RepoStructureExplorerTool,
    ExploreGraphStructure,
    ExploreTreeStructure, ExploreTreeStructure_simple,
    ExploreOneHopStructure, ExploreTwoHopStructure, ExploreThreeHopStructure, SPECIFIC_HOP_STRUCTURE_EXPLORERS
)
from utils.runtime.fragmental_content_tools import (
    ALL_FRAGMENTAL_CONTENT_TOOLS,
    GetFileContentTool,
    SearchClassTool,
    SearchFunctionTool,
    SearchKeywordSnippetsTool,
    SearchLineContextTool,
)
from utils.runtime.fragmental_structure_tools import (
    ALL_FRAGMENTAL_STRUCTURE_TOOLS,
    GetDirectoryStructureTool,
    GetFileStructuresTool,
    SearchClassStructuresTool,
    RetrieveClassAncestorsAndMethodsTool,
    ExploreClassHierarchyTool,
    SearchModuleRelTool,
)
# from utils.runtime.fragmental_structure_tools import (
    
# )
from utils.runtime.search_repo_tools import (
    SearchEntityTool,
    SearchRepoTool,
)

ALL_FUNCTIONS = [
                'explore_tree_structure', 'explore_repo_structure', 'explore_graph_structure', 
                'search_code_snippets',
                'get_entity_contents'
                ]
    
# from openhands.events.tool import ToolCallMetadata
logger = logging.getLogger()

SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>
"""

_FINISH_DESCRIPTION = """Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task."""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish',
        description=_FINISH_DESCRIPTION,
    ),
)


def combine_thought(action: Action, thought: str) -> Action:
    if hasattr(action, 'raw_content') and thought:
        action.raw_content = thought
    if hasattr(action, 'thought') and thought:
        action.thought += '\n' + thought
    return action


def response_to_actions(response: ModelResponse) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    assistant_msg = response.choices[0].message
    if assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(
                    f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                ) from e
            if tool_call.function.name == 'finish':
                if list(arguments.values()):
                    action = FinishAction(thought=list(arguments.values())[0])
                else:
                    action = FinishAction()
            
            elif tool_call.function.name == 'get_entity_contents':
                # We implement this in agent_skills, which can be used via Jupyter
                # logger.info(arguments)
                # converted_arguments = {'search_terms': arguments.get('entity_names', [])}
                # code = f'print(search_code_snippets(**{converted_arguments}))'
                code = f'print(get_entity_contents(**{arguments}))'
                logger.debug(f'TOOL CALL: get_entity_contents with code: {code}')
                action = IPythonRunCellAction(code=code,
                                              function_name=tool_call.function.name,
                                              tool_call_id=tool_call.id)  # include_extra=False
            elif tool_call.function.name in ALL_FRAGMENTAL_CONTENT_TOOLS:
                func_name = tool_call.function.name
                converted_arguments = {}
                if func_name == 'get_file_content':
                    converted_arguments['search_terms'] = [arguments.get('file_path', '')]
                elif func_name == 'search_class_content':
                    converted_arguments['search_terms'] = [arguments.get('module_name', '')]
                elif func_name == 'search_function_contents':
                    converted_arguments['search_terms'] = arguments.get('module_names', [])
                elif func_name == 'search_snippets_by_keywords':
                    converted_arguments['search_terms'] = arguments.get('search_terms', [])
                    converted_arguments['file_path_or_pattern'] = arguments.get('file_pattern', '**/*.py')
                elif func_name == 'search_context_by_lines':
                    converted_arguments['line_nums'] = arguments.get('line_nums', [])
                    converted_arguments['file_path_or_pattern'] = arguments.get('file_path', '')
                    
                # We implement this in agent_skills, which can be used via Jupyter
                code = f'print(search_code_snippets(**{converted_arguments}))'
                logger.debug(
                    f'TOOL CALL: search_code_snippets with code: {code}'
                )
                action = IPythonRunCellAction(code=code,
                                              function_name=tool_call.function.name,
                                              tool_call_id=tool_call.id)  # include_extra=False
            elif tool_call.function.name in SPECIFIC_HOP_STRUCTURE_EXPLORERS:
                func_name = tool_call.function.name
                hop_num = SPECIFIC_HOP_STRUCTURE_EXPLORERS.index(func_name)
                arguments['traversal_depth'] = hop_num
                code = f'print(explore_tree_structure(**{arguments}))'
                logger.debug(f'TOOL CALL: explore_tree_structure with code: {code}')
                action = IPythonRunCellAction(code=code,
                                              function_name=tool_call.function.name,
                                              tool_call_id=tool_call.id)
            elif tool_call.function.name in ALL_FRAGMENTAL_STRUCTURE_TOOLS:
                func_name = tool_call.function.name
                converted_arguments = {}
                if func_name == 'get_directory_structure':
                    converted_arguments['start_entities'] = [arguments.get('dir_path', '')]
                    converted_arguments['traversal_depth'] = arguments.get('traversal_depth', 2)
                    converted_arguments['dependency_type_filter'] = ['contains']
                    converted_arguments['entity_type_filter'] = ['file', 'directory']
                elif func_name == 'get_file_structures':
                    converted_arguments['start_entities'] = arguments.get('file_list', [])
                    converted_arguments['traversal_depth'] = 1
                    converted_arguments['dependency_type_filter'] = ['contains']
                    # converted_arguments['entity_type_filter'] = ['class', 'function']
                elif func_name == 'search_class_structures':
                    converted_arguments['start_entities'] = arguments.get('class_names', [])
                    converted_arguments['traversal_depth'] = 1
                    converted_arguments['dependency_type_filter'] = ['contains']
                    # converted_arguments['entity_type_filter'] = ['class', 'function']
                elif func_name == 'retrieve_class_ancestors_and_methods':
                    converted_arguments['start_entities'] = arguments.get('module_names', [])
                    converted_arguments['traversal_depth'] = -1
                    converted_arguments['dependency_type_filter'] = ['inherits', 'contains']
                elif func_name == 'explore_class_hierarchy':
                    converted_arguments['start_entities'] = arguments.get('module_names', [])
                    converted_arguments['traversal_depth'] = -1
                    converted_arguments['dependency_type_filter'] = ['inherits', 'contains']
                    converted_arguments['direction'] = 'both'
                elif func_name == 'search_invoke_and_reference':
                    converted_arguments['start_entities'] = arguments.get('module_names', [])
                    converted_arguments['traversal_depth'] = 1
                    converted_arguments['dependency_type_filter'] = ['invokes']
                    converted_arguments['direction'] = 'both'
                    
                code = f'print(explore_tree_structure(**{converted_arguments}))'
                action = IPythonRunCellAction(code=code,
                                              function_name=tool_call.function.name,
                                              tool_call_id=tool_call.id)
                
            elif tool_call.function.name in ALL_FUNCTIONS:
                # We implement this in agent_skills, which can be used via Jupyter
                func_name = tool_call.function.name
                code = f'print({func_name}(**{arguments}))'
                logger.debug(f'TOOL CALL: {func_name} with code: {code}')
                action = IPythonRunCellAction(code=code,
                                              function_name=func_name,
                                              tool_call_id=tool_call.id)  # include_extra=False
            else:
                raise RuntimeError(f'Unknown tool call: {tool_call.function.name}')

            # We only add thought to the first action
            if i == 0:
                action = combine_thought(action, thought)

            actions.append(action)
    else:
        actions.append(
            MessageAction(raw_content=assistant_msg.content, content=assistant_msg.content)
        )

    assert len(actions) >= 1
    return actions


def get_tools(
        # codeact_enable_cmd: bool = False,
        codeact_enable_search_keyword: bool = False,
        codeact_enable_search_entity: bool = False,
        codeact_enable_fragmental_content_tools: bool = False,
        
        # codeact_enable_search_call_relation: bool = False,
        # codeact_enable_explore_structure: bool = False,
        codeact_enable_tree_structure_traverser: bool = False,
        codeact_enable_specific_hops_structure_traverser: bool = False,
        codeact_enable_graph_structure_traverser: bool = False,
        codeact_enable_fragmental_structure_tools: bool = False,
        simple_desc: bool = False,
        
) -> list[ChatCompletionToolParam]:
    # tools = [CmdRunTool, FinishTool] # SOTA
    tools = [FinishTool]
    # if codeact_enable_cmd:
    #     tools.append(CmdRunTool)
    if codeact_enable_search_keyword:
        tools.append(SearchRepoTool)
    if codeact_enable_search_entity:
        tools.append(SearchEntityTool)
    if codeact_enable_fragmental_content_tools:
        tools.extend([
            GetFileContentTool,
            SearchClassTool,
            SearchFunctionTool,
            # SearchKeywordSnippetsTool,
            # SearchLineContextTool
        ])
    if codeact_enable_tree_structure_traverser:
        if simple_desc:
            tools.append(ExploreTreeStructure_simple)
        else:
            tools.append(ExploreTreeStructure)
    if codeact_enable_graph_structure_traverser:
        tools.append(ExploreGraphStructure)
    if codeact_enable_specific_hops_structure_traverser:
        tools.extend([
            ExploreOneHopStructure, ExploreTwoHopStructure, ExploreThreeHopStructure
        ])
    if codeact_enable_fragmental_structure_tools:
        tools.extend([
            GetDirectoryStructureTool, GetFileStructuresTool, SearchClassStructuresTool,
            # RetrieveClassAncestorsAndMethodsTool,
            # ExploreClassHierarchyTool,
            SearchModuleRelTool,
        ])
    # if codeact_enable_explore_structure:
    #     tools.append(RepoStructureExplorerTool)
    # if codeact_enable_search_call_relation:
    #     tools.append(SearchCallRelTool)
    
    return tools


