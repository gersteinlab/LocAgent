import logging
import tiktoken
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    "claude-instant-1": 0.00000163,
    "claude-2": 0.00001102,
    "claude-3-opus-20240229": 0.000015,
    "claude-3-sonnet-20240229": 0.000003,
    "claude-3-haiku-20240307": 0.00000025,
    "gpt-3.5-turbo-16k-0613": 0.0000015,
    "gpt-3.5-turbo-0613": 0.0000015,
    "gpt-3.5-turbo-1106": 0.000001,
    "gpt-35-turbo-0613": 0.0000015,
    "gpt-35-turbo": 0.0000015,  # probably still 0613
    "gpt-4-0613": 0.00003,
    "gpt-4-32k-0613": 0.00006,
    "gpt-4-32k": 0.00006,
    "gpt-4-1106-preview": 0.00001,
    "gpt-4-0125-preview": 0.00001,
    "gpt-4o": 0.000005,
    'azure/gpt-4o': 0.000005,
    'gpt-4o-2024-05-13': 0.000005,
    'openai/gpt-4o-2024-05-13': 0.000005,
    'openai/gpt-4o-mini-2024-07-18': 1.65e-07,
    'openai/deepseek-v2.5': 1.4e-07,
    'deepseek/deepseek-chat': 1.4e-07,
    'deepseek-ai/DeepSeek-R1': 5.5e-07,
    'litellm_proxy/claude-3-5-sonnet-20241022': 0.000003,
    'litellm_proxy/gpt-4o-2024-05-13': 0.000005,
    'litellm_proxy/o3-mini-2025-01-31': 1.1e-06,
    'azure/gpt-4o-mini-ft': 1.65e-07,
    'azure/gpt-4o-mini-1029-ft': 1.65e-07,
    'azure/gpt-4o-mini': 1.65e-07,
    'hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct': 0,
    'hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct': 0,
}

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    "claude-instant-1": 0.00000551,
    "claude-2": 0.00003268,
    "claude-3-opus-20240229": 0.000075,
    "claude-3-sonnet-20240229": 0.000015,
    "claude-3-haiku-20240307": 0.00000125,
    "gpt-3.5-turbo-16k-0613": 0.000002,
    "gpt-3.5-turbo-16k": 0.000002,
    "gpt-3.5-turbo-1106": 0.000002,
    "gpt-35-turbo-0613": 0.000002,
    "gpt-35-turbo": 0.000002,
    "gpt-4-0613": 0.00006,
    "gpt-4-32k-0613": 0.00012,
    "gpt-4-32k": 0.00012,
    "gpt-4-1106-preview": 0.00003,
    "gpt-4-0125-preview": 0.00003,
    "gpt-4o": 0.000015,
    'azure/gpt-4o': 0.000015,
    'openai/gpt-4o-2024-05-13': 0.000015,
    'gpt-4o-2024-05-13': 0.000015,
    'openai/gpt-4o-mini-2024-07-18': 6.6e-07,
    'openai/deepseek-v2.5': 2.8e-07,
    'deepseek/deepseek-chat': 2.8e-07,
    'deepseek-ai/DeepSeek-R1': 2.2e-06,
    'litellm_proxy/claude-3-5-sonnet-20241022': 0.000015,
    'litellm_proxy/gpt-4o-2024-05-13': 0.000015,
    'litellm_proxy/o3-mini-2025-01-31': 4.4e-06,
    'azure/gpt-4o-mini-ft': 6.6e-07,
    'azure/gpt-4o-mini-1029-ft': 6.6e-07,
    'azure/gpt-4o-mini': 6.6e-07,
    'hosted_vllm/Qwen/Qwen2.5-Coder-7B-Instruct': 0,
    'hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct': 0,
}

def calc_cost(model_name, input_tokens, output_tokens):
    """
    Calculates the cost of a response from the openai API.

    Args:
    response (openai.ChatCompletion): The response from the API.

    Returns:
    float: The cost of the response.
    """
    if model_name.split('/')[0] == 'hosted_vllm':
        return 0
    elif 'qwen' in model_name.lower():
        return 0
    
    cost = (
        MODEL_COST_PER_INPUT[model_name] * input_tokens
        + MODEL_COST_PER_OUTPUT[model_name] * output_tokens
    )
    logger.info(
        f"input_tokens={input_tokens}, output_tokens={output_tokens}, cost={cost:.2f}"
    )
    return cost

def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3 # every reply is primed with <|start|>assistant<|message|>
    tokens_per_name = 1

    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    num_tokens += tokens_per_message
    return num_tokens
