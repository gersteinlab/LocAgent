from dataclasses import dataclass
from pydantic import BaseModel, Field

class ActionTypeSchema(BaseModel):

    MESSAGE: str = Field(default='message')
    """Represents a message.
    """

    RUN_IPYTHON: str = Field(default='run_ipython')
    """Runs a IPython cell.
    """

    FINISH: str = Field(default='finish')
    """If you're absolutely certain that you've completed your task and have tested your work,
    use the finish action to stop working.
    """

ActionType = ActionTypeSchema()

@dataclass
class Action:
    raw_content: str = ''

# @dataclass
# class FunctionObject:
#     name: str
#     arguments: list

@dataclass
class MessageAction(Action):
    content: str = ''
    action_type: str = ActionType.MESSAGE
    

@dataclass
class FinishAction(Action):
    thought: str = ''
    action_type: str = ActionType.FINISH
    

@dataclass
class IPythonRunCellAction(Action):
    code: str = ''
    thought: str = ''
    function_name: str = ''
    tool_call_id: str = ''
    action_type: str = ActionType.RUN_IPYTHON
    # functions: list[FunctionObject] = Field(default_factory=list)

