import os
from pathlib import Path
from typing import Dict, Tuple

import yaml

DEFAULT_CONFIG: Dict[str, object] = {
    'model': 'nvidia_nim/qwen/qwen2-5-coder-32b-instruct',
    'temperature': 0.7,
    'max_iterations': 20,
    'api_key_env': 'NVIDIA_API_KEY',
    'nim_api_base': 'https://integrate.api.nvidia.com/v1',
    'fallback_model': 'openai/gpt-4o-mini',
    'use_function_calling': True,
    'simple_desc': False,
    'max_retries': 3,
    'retry_backoff_sec': 2,
}


def load_config(config_path: str | None = None) -> Tuple[Dict[str, object], str]:
    if not config_path:
        config_path = os.environ.get('BICAMERAL_CONFIG_PATH')

    if not config_path:
        config_path = str(Path(__file__).resolve().parents[1] / 'config.yaml')

    config = DEFAULT_CONFIG.copy()
    path = Path(config_path)
    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            config.update(data)
    return config, str(path)
