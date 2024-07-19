from dataclasses import dataclass, field
from typing import Any, Optional

from ..base import Instance
from ..components import LLM, LLMResult


# TODO: This needs a complete overhaul (lots of manual fiddling), if the goal is to get
#  useful prompt perplexity measures. If we decide not to do that (or we use some other
#  API), then we won't use transformers at all. (It was only useful for Mistral/Gemma.)

@dataclass
class TransformersConfig:
    # RNG seed for replication of results.
    seed: int = 314159

    # The prompt for system instructions (or None for LLMs that don't support one).
    system_prompt: Optional[str] = None

    # Whether to use the transformers.Pipeline chat templating functionality.
    use_chat_template: bool = False

    # See transformers.AutoModelForCausalLM.from_pretrained for details.
    # NOTE: Skip 'quantization_config', which is handled specially.
    model_params: dict[str, Any] = field(
        default_factory=lambda: {"trust_remote_code": True},
    )

    # See transformers.GenerationConfig for details.
    generation_params: dict[str, Any] = field(
        default_factory=lambda: {
            "return_full_text": True,
            "max_new_tokens": 5,
            "num_return_sequences": 1,
            "output_scores": True,
            "return_dict_in_generate": True,
        },
    )


class TransformersLLM(LLM):
    """Interface for all LLMs using the HuggingFace transformers API."""

    def __init__(self, model_name: str, cfg: TransformersConfig):
        # Delayed imports.
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            set_seed,
        )

        # Basic initialization.
        set_seed(cfg.seed)
        super().__init__(model_name)
        self.cfg = cfg

        # Quantization.
        params = {**self.cfg.model_params}  # Convert OmegaConf -> dict

        # Pipeline initialization.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.cfg.model_params,
        )

    def __call__(self, text: str, instance: Instance, *args, **kwargs) -> LLMResult:
        if self.cfg.use_chat_template:
            if self.cfg.system_prompt is None:
                prompt = [{"role": "user", "content": text}]
            else:
                prompt = [
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": text},
                ]
        else:
            prompt = text
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **self.cfg.generation_params)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        #for tok, score in zip(generated_tokens[0], transition_scores[0]):
        #    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
        #generated_text = output[0]["generated_text"]
        #return LLMResult(generated_text)
        raise NotImplementedError
