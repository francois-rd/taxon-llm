from dataclasses import dataclass
from typing import Optional


InstanceId = str
QAId = str
TemplateId = str

Case = int
RelationType = str

Label = str
Term = str


@dataclass
class Pairing:
    """
    Dataclass to store pairing data.

    id: Identifier of the pairing template (which has the same ID in each tree).
    flip_negative: Whether the positive/negative variation logic needs to be flipped.
    skill_type: The reasoning skill type of the relation of the pairing template.
    """
    id: TemplateId
    flip_negative: bool
    relation_type: RelationType


@dataclass
class Template:
    """
    Dataclass to store Template/Statement data.

    surface_form: Format string for the surface (i.e., text) form of the statement.
    source_term: String value of the source term.
    target_term: String value of the target term.
    relation_type: The skill type of the relation of the statement.
    """
    surface_form: str
    source_term: Term
    target_term: Term
    relation_type: RelationType


@dataclass
class Instance:
    """
    Dataclass to store all instance data and meta-data.

    id: Identifier for this instance within the data subset.
    qa_id: Identifier of the QA instance from the base dataset.
    question: Question text of the QA instance.
    answer_choices: Answer choices of the QA instance; form: {label: term}.
    label: Chosen answer label (which can be different from the QA instance label
        due to anti-factual statement logic construction).
    pairing: Pairing data. None when no statements are given.
    reduction_cases: Indicates which statements reduce to which cases when dominated by
        the 'pairing'; form {template_id: case_id}. None when no statements are given.
    statements: All Template/Statement data for each statement in each tree; form:
            {tree_label: {template_id: Template}}.
    statements_order: Ordered list of all Templates/Statements from all trees; form:
            [(tree_label, statement_id)].
    """
    id: InstanceId
    qa_id: QAId
    question: str
    answer_choices: dict[Label, Term]
    label: Label
    pairing: Optional[Pairing]
    reduction_cases: Optional[dict[TemplateId, Case]]
    statements: dict[Label, dict[TemplateId, Template]]
    statements_ordering: list[tuple[Label, TemplateId]]

    def tree_size(self) -> int:
        arbitrary_tree = next(self.statements.values().__iter__())
        return len(arbitrary_tree)

    def reasoning_hops(self) -> int:
        return 0 if self.reduction_cases is None else 1 + len(self.reduction_cases)

    def distractors(self) -> int:
        return self.tree_size() - self.reasoning_hops()

    def is_pairing(self, template_id: TemplateId) -> bool:
        return self.pairing is not None and template_id == self.pairing.id

    def is_in_reduction_cases(self, template_id: TemplateId) -> bool:
        return self.reduction_cases is not None and template_id in self.reduction_cases

    def is_reasoning_hop(self, template_id: TemplateId) -> bool:
        return self.is_pairing(template_id) or self.is_in_reduction_cases(template_id)

    def is_distractor(self, template_id: TemplateId) -> bool:
        return not self.is_reasoning_hop(template_id)
