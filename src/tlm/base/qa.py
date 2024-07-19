from dataclasses import dataclass

from .instance import Label, QAId


@dataclass
class QAData:
    """
    Contains all relevant data for a single QA instance from the underlying QA dataset.

    id: A unique identifier differentiating this QA instance from others.
    factual_label: The factually correct answer label for this QA instance.
    """

    id: QAId
    factual_label: Label
