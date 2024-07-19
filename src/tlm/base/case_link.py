from dataclasses import dataclass

from .instance import Case, RelationType


@dataclass
class CaseLink:
    """
    A permutation case linking two statements based on their relations' skill type.
    Typically, one statement will dominate over the other.

    The five permutation cases are:
        Case 0: A relation1 B; C relation2 D (no linking relation)
        Case 1: A relation1 B; B relation2 C (relation1.target == relation2.source)
        Case 2: A relation1 B; C relation2 B (relation1.target == relation2.target)
        Case 3: B relation1 A; B relation2 C (relation1.source == relation2.source)
        Case 4: B relation1 A; C relation2 B (relation1.source == relation2.target)

    NOTE: The cases are equivalent with respect to permutation of the order of the
    statements unless the relation types are the same (where the surface form of the
    subsumed statement can sometimes be different). In general, however:
        Case 1 permutes to case 4
        Case 4 permutes to case 1
        All other cases permute to themselves

    r1_type: Relation skill type of the dominant statement
    r2_type: Relation skill type of the subsumed statement
    case: Permutation case linking the two statements
    """
    r1_type: RelationType
    r2_type: RelationType
    case: Case

    def equivalent(self) -> "CaseLink":
        if self.case == 1:
            return CaseLink(self.r2_type, self.r1_type, 4)
        elif self.case == 0 or self.case == 2 or self.case == 3:
            return CaseLink(self.r2_type, self.r1_type, self.case)
        elif self.case == 4:
            return CaseLink(self.r2_type, self.r1_type, 1)
        else:
            raise TypeError(f"Unsupported permutation case: {self.case}")

    def as_tuple(self) -> tuple[RelationType, RelationType, Case]:
        return self.r1_type, self.r2_type, self.case
