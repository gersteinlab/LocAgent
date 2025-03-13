import re
from typing import Union, List
from utils.actions.action import Action, IPythonRunCellAction, FinishAction, MessageAction, ActionType
from utils.runtime.function_calling import response_to_actions as parse_tool_calls_to_actions
import logging


class ResponseParser:
    def __init__(self):
        # Need pay attention to the item order in self.action_parsers
        # super().__init__()

        self.action_parsers = [
            CodeActActionParserFinish(),
            # CodeActActionParserCmdRun(),
            CodeActActionParserIPythonRunCell(),
            # CodeActActionParserAgentDelegate(),
        ]
        self.default_parser = CodeActActionParserMessage()

    def parse(self, response) -> Union[List[Action], Action]:
        # using tool calling
        if response.choices[0].message.tool_calls:
            try:
                actions = parse_tool_calls_to_actions(response)
                return actions
            except:
                logging.info("Un know tools")
        
        # using code to call tools
        action_str = self.parse_response(response)
        return self.parse_action(action_str)

    def parse_response(self, response) -> str:
        action = response.choices[0].message.content
        if action is None:
            return ''
        for lang in ['bash', 'ipython', 'browse']:
            if f'<execute_{lang}>' in action and f'</execute_{lang}' in action and f'</execute_{lang}>' not in action:
                action += '>'
            if f'<execute_{lang}>' in action and f'</execute_{lang}>' not in action:
                action += f'</execute_{lang}>'
        return action

    def parse_action(self, action_str: str) -> Union[List[Action], Action]:
        for action_parser in self.action_parsers:
            if action_parser.check_condition(action_str):
                return action_parser.parse(action_str)
        return self.default_parser.parse(action_str)


class CodeActActionParserMessage:
    """Parser action:
    - MessageAction(content) - Message action to run (e.g. ask for clarification)
    """

    def __init__(
        self,
    ):
        pass

    def check_condition(self, action_str: str) -> bool:
        # We assume the LLM is GOOD enough that when it returns pure natural language
        # it wants to talk to the user
        return True

    def parse(self, action_str: str) -> Action:
        action = MessageAction(raw_content=action_str, content=action_str)
        return action


class CodeActActionParserIPythonRunCell:
    def __init__(self):
        self.commands = []

    def check_condition(self, action_str: str) -> bool:
        python_code = re.search(
            r'<execute_ipython>(.*?)</execute_ipython>', action_str, re.DOTALL
        )
        return python_code is not None

    def parse(self, action_str: str) -> str:
        """
        Extracts and stores the commands within the <execute_ipython> tags.
        """
        python_code = re.search(
            r'<execute_ipython>(.*?)</execute_ipython>', action_str, re.DOTALL
        )
        assert (
            python_code is not None
        ), 'python_code should not be None when parse is called'
        
        code_group = python_code.group(1).strip()
        thought = action_str.replace(python_code.group(0), '').strip()

        action = IPythonRunCellAction(raw_content=action_str, code=code_group, thought=thought)
        
        # for code in code_group.split('\n'):
        #     func_name, args = self.extract_function(code)
        #     if func_name:
        #         action.functions.append(FunctionObject(func_name, args))
        
        return action
    # def parse(self, action_str: str) -> list:
    #     """
    #     Extracts and stores the commands within all <execute_ipython> tags.
    #     """
    #     # Find all occurrences of <execute_ipython>...</execute_ipython>
    #     python_code_blocks = re.findall(
    #         r'<execute_ipython>(.*?)</execute_ipython>', action_str, re.DOTALL
    #     )
    #     assert (
    #         len(python_code_blocks) > 0
    #     ), 'python_code should not be None when parse is called'
        
    #     # Store each code block and clean up leading/trailing whitespace
    #     self.commands = [code_block.strip() for code_block in python_code_blocks]

    #     # Remove the <execute_ipython> blocks from the original string to get the thought
    #     thought = re.sub(r'<execute_ipython>.*?</execute_ipython>', '', action_str, flags=re.DOTALL).strip()
        
    #     action = IPythonRunCellAction(raw_content=action_str, code=self.commands, thought=thought)
    #     return action

    def extract_function(self, code_str):
        """
        Extracts the function name and arguments from a string like `open_file('app.py')`.
        """
        func_pattern = r"(\w+)\((.*)\)"
        match = re.match(func_pattern, code_str)
        if match:
            func_name = match.group(1)
            # Evaluate the arguments safely (this assumes simple literals like strings, numbers, etc.)
            args = eval(f"[{match.group(2)}]") if match.group(2) else []
            return func_name, args
        return None, None

class CodeActActionParserFinish:
    def check_condition(self, action_str: str) -> bool:
        self.finish_command = re.search(r'<finish>.*</finish>', action_str, re.DOTALL)
        return self.finish_command is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.finish_command is not None
        ), 'self.finish_command should not be None when parse is called'
        thought = action_str.replace(self.finish_command.group(0), '').strip()
        return FinishAction(raw_content=action_str, thought=thought)



