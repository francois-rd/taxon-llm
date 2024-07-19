from typing import Generator
from pathlib import Path
import os

from tqdm import tqdm

from .configs import AnalysisLoaderConfig, GeneralConfig, ResourcesConfig, update
from ..base import CaseLink, Instance, QAData
from ..components import (
    AnalysisBase,
    AnswerLabels,
    LLMOutputParser,
    LLMResult,
    SurfaceForms,
)
from ..io import ensure_path, load_dataclass_jsonl, load_jsonl, load_records_csv


def surface_forms_loader(file_path: str, **kwargs) -> SurfaceForms:
    """
    Creates a Reducer, registering RelationalCaseLinks from a formatted reductions file.
    loaded from the given file path. Relations should match the relation types in the
    reductions file. kwargs are passed to pandas.read_csv().
    """
    forms = SurfaceForms()
    for data in load_records_csv(file_path, **kwargs):
        c = CaseLink(data["relation1_or_pairing"], data["relation2"], int(data["case"]))
        forms.register(c, data["relation2_surface_form"])
    return forms


def dataset_loader(resources: ResourcesConfig, general: GeneralConfig) -> Generator:
    """
    Very simple checkpoint management around loading of a dataset. Returns a Generator
    that loads a dataset of Instances, yields each Instance in turn, and then (nearly)
    atomically creates a file with the Instance's identifier as file name to a
    checkpointing folder. Then, later re-runs from the same checkpointing folder cause
    those Instances whose identifiers have been written to file to be skipped.
    """
    def _generator():
        yielded = False
        dataset = load_dataclass_jsonl(resources.dataset_file, Instance)
        for instance in tqdm(dataset, desc="Progress", disable=not general.verbose):
            instance_file = os.path.join(resources.checkpoint_dir, instance.id)
            if os.path.isfile(instance_file):
                continue
            else:
                yield instance
                yielded = True
                Path(ensure_path(instance_file)).touch()
        if not yielded and general.verbose:
            print(f"No yield. Forgot to set new job ID? Job ID={resources.job_id}")

    return _generator()


def csqa_loader(r: ResourcesConfig) -> list[QAData]:
    """Loads the base CSQA data, returning a mapping between its IDs and labels."""
    return [QAData(d["id"], d["answerKey"]) for d in load_jsonl(r.csqa_raw_dev_file)]


def analysis_loader(
    resources: ResourcesConfig,
    loader_cfg: AnalysisLoaderConfig,
    qa_dataset: list[QAData],
    llm_parser: LLMOutputParser,
    *args,
    **kwargs,
) -> list[AnalysisBase]:
    """
    Utility to load data for performing analysis. Specifically, collates LLM
    performance data from all tree sizes and all LLMs into one data structure.
    """
    analyses = []
    qa_dataset = {qa_data.id: qa_data for qa_data in qa_dataset}
    for tree_size in loader_cfg.tree_sizes:
        with update(resources, tree_size=tree_size) as r1:
            dataset = load_dataclass_jsonl(r1.dataset_file, Instance)
        dataset = {instance.id: instance for instance in dataset}
        for llm in loader_cfg.llms:
            num_fails = 0
            with update(r1, tree_size=tree_size, llm=llm) as r2:
                llm_results = load_dataclass_jsonl(r2.llm_results_file, LLMResult)
            for result in llm_results:
                try:
                    instance = dataset[result.instance_id]
                except KeyError:
                    num_fails += 1
                    continue
                qa_data = qa_dataset[instance.qa_id]
                label = llm_parser(result.generated_text, instance, *args, **kwargs)
                labels = AnswerLabels(label, instance.label, qa_data.factual_label)
                a_b = AnalysisBase(tree_size, llm, instance, qa_data, result, labels)
                analyses.append(a_b)
            if num_fails > 0:
                print(f"Num fails for {llm} on tree size {tree_size}: {num_fails}")
    return analyses
