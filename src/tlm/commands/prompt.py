from dataclasses import dataclass
from typing import Iterable

from .configs import GeneralConfig, InstanceSurfacerConfig, ResourcesConfig
from .loaders import surface_forms_loader
from ..base import Instance
from ..components import (
    InstanceSurfacer,
    LLM,
    OrderingSurfacer,
    QADataSurfacer,
    TemplateSurfacer,
    TermSurfacer,
    TextSurfacer,
)
from ..io import save_dataclass_jsonl


@dataclass
class PseudoConfig:
    dataset: Iterable[Instance]
    llm: LLM


class Command:
    """Prompts a specific LLM with data corresponding to a specific tree_size."""

    def __init__(
        self,
        resources: ResourcesConfig,
        general: GeneralConfig,
        surfacer_cfg: InstanceSurfacerConfig,
        pseudo_cfg: PseudoConfig,
    ):
        self.resources = resources
        self.general = general
        self.surfacer_cfg = surfacer_cfg
        self.pseudo_cfg = pseudo_cfg
        self.surfacer = self._init_surfacer()

    def _init_surfacer(self) -> InstanceSurfacer:
        # Aliases to shorten line lengths.
        qa_surfacer = self.surfacer_cfg.qa_data_surfacer

        # Create OrderingSurfacer only if trees are given.
        ordering_surfacer = None
        if self.resources.tree_size > 0:
            ordering_surfacer = self._init_ordering_surfacer()

        # Create Surfacer.
        return InstanceSurfacer(
            prefix=self.surfacer_cfg.prefix,
            surfacer_separator=self.surfacer_cfg.surfacer_separator,
            prefix_surfacer=TextSurfacer(
                prefix=self.surfacer_cfg.prefix_surfacer.prefix,
                text=self.surfacer_cfg.prefix_surfacer.text,
            ),
            ordering_surfacer=ordering_surfacer,
            qa_data_surfacer=QADataSurfacer(
                prefix=qa_surfacer.prefix,
                question_answer_separator=qa_surfacer.question_answer_separator,
                answer_choice_separator=qa_surfacer.answer_choice_separator,
                answer_choice_formatter=qa_surfacer.answer_choice_formatter,
                omit_question=qa_surfacer.omit_question,
            ),
            suffix_surfacer=TextSurfacer(
                prefix=self.surfacer_cfg.suffix_surfacer.prefix,
                text=self.surfacer_cfg.suffix_surfacer.text,
            ),
        )

    def _init_ordering_surfacer(self):
        # Aliases to shorten line lengths.
        surfacer_cfg = self.surfacer_cfg.ordering_surfacer

        # Create Surfacer.
        return OrderingSurfacer(
            prefix=surfacer_cfg.prefix,
            template_separator=surfacer_cfg.template_separator,
            template_surfacer=TemplateSurfacer(
                prefix=surfacer_cfg.template_surfacer.prefix,
                term_surfacer=TermSurfacer(
                    prefix=surfacer_cfg.template_surfacer.term_surfacer.prefix,
                    suffix=surfacer_cfg.template_surfacer.term_surfacer.suffix,
                ),
                forms=surface_forms_loader(
                    file_path=self.resources.reductions_file,
                    comment=self.general.csv_line_comment,
                ),
            ),
        )

    def run(self):
        results = []
        for instance in self.pseudo_cfg.dataset:
            results.append(self.pseudo_cfg.llm(self.surfacer(instance), instance))
        save_dataclass_jsonl(self.resources.llm_results_file, *results)
        if self.general.verbose:
            print("Done.")
