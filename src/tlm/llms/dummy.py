from dataclasses import dataclass

from ..base import Instance
from ..components import LLM, LLMResult


@dataclass
class DummyConfig:
    # Dummy response (prediction label) to give to all model calls.
    response: str = "A"


class DummyLLM(LLM):
    """
    A fake/dummy LLM that outputs a dummy response to all calls. Useful for debugging
    as well as recording benchmark snapshots.
    """

    def __init__(self, model_name: str, cfg: DummyConfig):
        super().__init__(model_name)
        self.response = cfg.response

    def __call__(self, _: str, instance: Instance, *args, **kwargs) -> LLMResult:
        return LLMResult(instance.id, self.response)
