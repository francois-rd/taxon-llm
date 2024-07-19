from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Optional


@dataclass
class GeneralConfig:
    # Whether commands should operate verbosely.
    verbose: bool = False

    # General random seed for a command to be reproducible.
    random_seed: int = 314159

    # When loading CSV files, lines starting with this string are skipped.
    csv_line_comment: str = "#"


@dataclass
class ResourcesConfig:
    # Top level directory for resources, data, and results.
    root_dir: str = "data"

    # Checkpointing directory.
    checkpoint_dir: str = "${root_dir}/checkpoints/${job_id}"
    job_id: int = -1  # To override at runtime.

    # Input dataset/database directories.
    dataset_dir: str = "${root_dir}/${dataset_name}"
    term_database_dir: str = "${root_dir}/${term_database_name}"
    dataset_name: str = ""
    term_database_name: str = ""
    dataset_file: str = "${dataset_dir}/${dataset_name}_${tree_size}.jsonl"
    reductions_file: str = "${dataset_dir}/reductions.csv"

    # Specific dataset paths.
    csqa_raw_dev_file: str = "${root_dir}/csqa/dev_rand_split.jsonl"

    # Directories for all results.
    result_dir: str = "${root_dir}/results"

    # LLMResult directory and data.
    llm_results_dir: str = "${result_dir}/llm_results/${llm}"
    llm_results_file: str = "${llm_results_dir}/llm_results_${tree_size}.jsonl"

    # Analysis results directories and data.
    analysis_root_dir: str = "${result_dir}/analysis"
    basic_analysis_dir: str = "${analysis_root_dir}/basic"
    basic_analysis_data_file: str = "${basic_analysis_dir}/analysis.jsonl"
    basic_analysis_pretty_file: str = "${basic_analysis_dir}/analysis_{}_{}.pdf"
    graph_analysis_dir: str = "${analysis_root_dir}/graph"
    graph_analysis_data_file: str = "${graph_analysis_dir}/analysis.jsonl"
    graph_analysis_pretty_file: str = "${graph_analysis_dir}/analysis_{}.pdf"

    # Size of tree (number of Templates).
    tree_size: int = 2

    # Language model to use.
    llm: str = "dummy"


@contextmanager
def update(
    resources: ResourcesConfig,
    tree_size: Optional[int] = None,
    llm: Optional[str] = None,
):
    """Temporarily updates a ResourcesConfig's tree_size or llm value."""
    old_tree_size = None
    if tree_size is not None:
        old_tree_size = resources.tree_size
        resources.tree_size = tree_size
    old_llm = None
    if llm is not None:
        old_llm = resources.llm
        resources.llm = llm
    try:
        yield resources
    finally:
        if tree_size is not None:
            resources.tree_size = old_tree_size
        if llm is not None:
            resources.llm = old_llm


@dataclass
class AnalysisLoaderConfig:
    # Which tree sizes to include in the analysis.
    tree_sizes: list[int] = field(default_factory=list)

    # Which LLMs to include in the analysis.
    llms: list[str] = field(default_factory=list)


@dataclass
class SurfacerConfig:
    """See accord.components.surfacer.Surfacer for details."""

    prefix: str = ""


@dataclass
class TextSurfacerConfig(SurfacerConfig):
    """See accord.components.surfacer.TextSurfacer for details."""

    text: str = ""


@dataclass
class TermSurfacerConfig(SurfacerConfig):
    """See accord.components.surfacer.TermSurfacer for details."""

    suffix: str = ""


@dataclass
class TemplateSurfacerConfig(SurfacerConfig):
    """See accord.components.surfacer.TemplateSurfacer for details."""

    term_surfacer: TermSurfacerConfig = field(
        default_factory=lambda: TermSurfacerConfig("[", "]")
    )


@dataclass
class OrderingSurfacerConfig(SurfacerConfig):
    """See accord.components.surfacer.OrderingSurfacer for details."""

    template_separator: str = "\n"
    template_surfacer: TemplateSurfacerConfig = field(
        default_factory=lambda: TemplateSurfacerConfig("- ")
    )


@dataclass
class QADataSurfacerConfig(SurfacerConfig):
    """See accord.components.surfacer.QADataSurfacer for details."""

    question_answer_separator: str = "\n"
    answer_choice_separator: str = "    "
    answer_choice_formatter: str = "{}: {}"
    omit_question: bool = False


@dataclass
class InstanceSurfacerConfig(SurfacerConfig):
    """See accord.components.surfacer.InstanceSurfacer for details."""

    surfacer_separator: str = "\n"
    prefix_surfacer: TextSurfacerConfig = field(
        default_factory=lambda: TextSurfacerConfig("Instructions:\n")
    )
    ordering_surfacer: OrderingSurfacerConfig = field(
        default_factory=lambda: OrderingSurfacerConfig("Statements:\n")
    )
    qa_data_surfacer: QADataSurfacerConfig = field(
        default_factory=lambda: QADataSurfacerConfig("Question:\n")
    )
    suffix_surfacer: TextSurfacerConfig = field(
        default_factory=lambda: TextSurfacerConfig("Answer:\n")
    )
