from dataclasses import dataclass, field
from typing import Any, Iterable
from enum import Enum
import math


from ...configs import GeneralConfig, ResourcesConfig
from ....base import Instance, RelationType
from ....components import AnalysisBase, safe_div
from ....io import save_dataclass_jsonl


class BinType(Enum):
    """The datatype of a given histogram bin."""

    TREE_SIZE = "TREE_SIZE"
    REASONING_HOPS = "REASONING_HOPS"
    DISTRACTORS = "DISTRACTORS"


@dataclass
class AnalysisResults:
    """Class for storing all analysis results in a table-ready format."""

    llm: str
    bin_type: BinType
    skill_type: RelationType
    skill_count: int
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

    # Which reasoning skill types to use, and in which order to produce them.
    skill_types: list[RelationType] = field(default_factory=list)

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
                if not properties["is_baseline"] and properties["skill_count"] == 0:
                    # Skip absence of a skill in the tree, except baseline.
                    continue
                if len(analyses) < self.cfg.group_size_cutoff:
                    # Skip small group sizes to avoid acc/se issues.
                    continue
                acc = self._accuracy(analyses, properties["is_baseline"])
                results.append(AnalysisResults(bin_type=bin_type, **properties, **acc))
        save_dataclass_jsonl(self.resources.basic_analysis_data_file, *results)

    def _group_analyses(self, bin_type: BinType) -> Iterable[Grouping]:
        results = {}
        for a in self.pseudo_cfg.analyses:
            result = results.setdefault(a.tree_size == 0, {}).setdefault(a.llm, {})
            for skill, count in self._skill_counts(a.instance, bin_type).items():
                result.setdefault(skill, {}).setdefault(count, []).append(a)
        for is_baseline, baseline_dict in results.items():
            for llm, skill_dict in baseline_dict.items():
                for skill_type, count_dict in skill_dict.items():
                    for skill_count, analyses in count_dict.items():
                        properties = {
                            "is_baseline": is_baseline,
                            "llm": llm,
                            "skill_type": skill_type,
                            "skill_count": skill_count,
                        }
                        yield properties, analyses

    def _skill_counts(
        self,
        instance: Instance,
        bin_type: BinType,
    ) -> dict[RelationType, int]:
        arbitrary_tree = next(instance.statements.values().__iter__())
        results = {skill_type: 0 for skill_type in self.cfg.skill_types}
        for template_id, template in arbitrary_tree.items():
            if bin_type == BinType.TREE_SIZE:
                results[template.relation_type] += 1
            elif bin_type == BinType.REASONING_HOPS:
                if instance.is_reasoning_hop(template_id):
                    results[template.relation_type] += 1
            elif bin_type == BinType.DISTRACTORS:
                if instance.is_distractor(template_id):
                    results[template.relation_type] += 1
            else:
                raise ValueError(f"Unsupported BinType: {bin_type}")
        return results

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
