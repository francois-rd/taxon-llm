from dataclasses import dataclass, field
from typing import Any, Optional
import os

from retry import retry

from ..base import Instance
from ..components import LLM, LLMResult


@dataclass
class OpenAIConfig:
    # The environment variable in which the OpenAI API key is stored (or None for
    # local VLLM server requests).
    api_env_var: Optional[str] = None

    # The file in which the base URL for accessing the model is stored (or None for
    # remote OpenAI server requests).
    base_url_file: Optional[str] = None

    # Whether to use the OpenAI Completions or OpenAI ChatCompletions protocol.
    use_chat: bool = True

    # The prompt for system instructions (or None for LLMs that don't support one).
    system_prompt: Optional[str] = None

    # See openai.OpenAI.completions.create or openai.OpenAI.chat.completions.create for
    # details. Skip 'model', 'messages', and 'prompt', which are handled specially.
    # Other options may exist when servicing requests based on VLLM servers.
    query_params: dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 5,
            "seed": 314159,
            "logprobs": True,
            "top_logprobs": 20,
        },
    )


class OpenAILLM(LLM):
    """
    Interface for all LLMs using the OpenAI API. Supports both requests to OpenAI
    servers (remote) and requests to a VLLM server (local).
    """

    def __init__(self, model_name: str, cfg: OpenAIConfig):
        from openai import OpenAI  # Delayed import.

        super().__init__(model_name)
        self.cfg = cfg
        api_key = "EMPTY" if cfg.api_env_var is None else os.environ[cfg.api_env_var]
        base_url = None
        if cfg.base_url_file is not None:
            with open(cfg.base_url_file) as f:
                base_url = f.read().strip()
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @retry(delay=1)
    def __call__(self, text: str, instance: Instance, *args, **kwargs) -> LLMResult:
        if self.cfg.use_chat:
            if self.cfg.system_prompt is None:
                messages = [{"role": "user", "content": text}]
            else:
                messages = [
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": text},
                ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.cfg.query_params,
            )
            generated_text = response.choices[0].message.content
        else:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=text,
                **self.cfg.query_params,
            )
            generated_text = response.choices[0].text
        logprobs = response.choices[0].logprobs
        return LLMResult(instance.id, generated_text, logprobs)
