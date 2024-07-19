from dataclasses import dataclass, field
from typing import Any, Iterable
from enum import Enum
import math

import networkx as nx

from ...configs import GeneralConfig, ResourcesConfig
from ....base import Instance
from ....components import AnalysisBase, safe_div
from ....io import save_dataclass_jsonl


class BinType(Enum):
    """The datatype of a given histogram bin."""

    MAX_DEGREE = "MAX_DEGREE"


@dataclass
class AnalysisResults:
    """Class for storing all analysis results in a table-ready format."""

    tree_size: int
    llm: str
    bin_type: BinType
    bin_value: int
    f_acc: float
    f_se: float  # -1 when is_baseline is True
    af_acc: float  # -1 when is_baseline is True
    af_se: float  # -1 when is_baseline is True
    is_baseline: bool


# Data format from grouping of analyses.
GroupProperties = dict[str, Any]
Grouping = tuple[GroupProperties, list[AnalysisBase]]


@dataclass
class Config:
    # Which BinTypes to use, and in which order to produce them.
    bin_types: list[BinType] = field(default_factory=list)

    # Groups smaller than this cutoff are ignored (to prevent accuracy/se issues).
    group_size_cutoff: int = 0


@dataclass
class PseudoConfig:
    analyses: list[AnalysisBase]


class Command:
    def __init__(
        self,
        resources: ResourcesConfig,
        _: GeneralConfig,
        cfg: Config,
        pseudo_cfg: PseudoConfig,
    ):
        self.resources = resources
        self.cfg = cfg
        self.pseudo_cfg = pseudo_cfg

    def run(self):
        results = []
        for bin_type in self.cfg.bin_types:
            for properties, analyses in self._group_analyses(bin_type):
                if len(analyses) < self.cfg.group_size_cutoff:
                    # Skip small group sizes to avoid acc/se issues.
                    continue
                acc = self._accuracy(analyses, properties["is_baseline"])
                results.append(AnalysisResults(bin_type=bin_type, **properties, **acc))
        save_dataclass_jsonl(self.resources.graph_analysis_data_file, *results)

    def _group_analyses(self, bin_type: BinType) -> Iterable[Grouping]:
        results = {}
        for a in self.pseudo_cfg.analyses:
            ts = a.tree_size
            r = results.setdefault(ts == 0, {}).setdefault(ts, {}).setdefault(a.llm, {})
            r.setdefault(self._bin_value(a.instance, bin_type, ts == 0), []).append(a)
        for is_baseline, baseline_dict in results.items():
            for tree_size, tree_size_dict in baseline_dict.items():
                for llm, llm_dict in tree_size_dict.items():
                    for bin_value, analyses in llm_dict.items():
                        properties = {
                            "is_baseline": is_baseline,
                            "tree_size": tree_size,
                            "llm": llm,
                            "bin_value": bin_value,
                        }
                        yield properties, analyses

    @staticmethod
    def _bin_value(instance: Instance, bin_type: BinType, is_baseline: bool) -> int:
        if is_baseline:
            return -1
        arbitrary_tree = next(instance.statements.values().__iter__())
        graph, variables = nx.Graph(), {}
        for t in arbitrary_tree.values():
            if t.source_term not in variables:
                variables[t.source_term] = len(variables)
            if t.target_term not in variables:
                variables[t.target_term] = len(variables)
            graph.add_edge(variables[t.source_term], variables[t.target_term])
        if bin_type == BinType.MAX_DEGREE:
            return max(degree for _, degree in graph.degree())
        else:
            raise ValueError(f"Unsupported BinType: {bin_type}")

    @staticmethod
    def _accuracy(analyses: list[AnalysisBase], is_baseline: bool) -> dict[str, float]:
        def acc(condition):
            score, count = 0, 0
            for analysis in analyses:
                if condition(analysis.labels.chosen, analysis.labels.factual):
                    if analysis.labels.generated == analysis.labels.chosen:
                        score += 1
                    count += 1
            prop = safe_div(score, count)
            wald_err = math.sqrt(safe_div(prop * (1 - prop), count, zero_div=0))
            return prop, wald_err

        def f_acc():
            return acc(lambda x, y: x == y)

        def af_acc():
            return acc(lambda x, y: x != y)

        f = [f_acc()[0], -1.0] if is_baseline else f_acc()
        af = [-1.0, -1.0] if is_baseline else af_acc()
        return dict(zip(["f_acc", "f_se", "af_acc", "af_se"], f + af))
