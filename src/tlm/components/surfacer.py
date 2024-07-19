from typing import Optional
import re

from ..base import CaseLink, Instance


class SurfaceForms:
    """
    Keeps track of surface form variations of subsumed relations in a CaseLink.
    """
    def __init__(self):
        self.permutations = {}

    def register(self, case_link: CaseLink, subsumed_surface_form: str):
        """Registers a CaseLink and its associated subsumed surface form."""
        self.permutations[case_link.as_tuple()] = subsumed_surface_form

    def get(self, case_link: CaseLink) -> Optional[str]:
        """
        Returns the surface form for a CaseLink. Returns the surface form of the
        equivalent CaseLink under case permutation if no surface form is registered
        for the given CaseLink. Returns None if neither the given nor the equivalent
        CaseLink have registered surface forms.
        """
        equiv = case_link.equivalent().as_tuple()
        if case_link.as_tuple() in self.permutations:
            return self.permutations[case_link.as_tuple()]
        elif equiv in self.permutations:
            return self.permutations[equiv]
        else:
            return None


class Surfacer:
    """
    Base class for Surfacers. A Surfacer surfaces (i.e., converts from a data structure
    format to plain text) some or all parts of a given Instance.
    """

    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, instance: Instance, *args, **kwargs) -> str:
        raise NotImplementedError


class TextSurfacer(Surfacer):
    """
    A TextSurfacer surfaces its prefix followed by some fixed text, regardless of
    the given Instance (which is effectively ignored).
    """

    def __init__(self, prefix: str, text: str):
        super().__init__(prefix)
        self.text = text

    def __call__(self, *args, **kwargs) -> str:
        return self.prefix + self.text


class TermSurfacer(Surfacer):
    """
    A TermSurfacer surfaces its prefix, followed by a Term (where any '_' is replaced
    with ' '), followed by its suffix. The term should be given as a kwarg ('term') to
    __call__(). The Instance is ignored.
    """

    def __init__(self, prefix: str, suffix: str):
        super().__init__(prefix)
        self.suffix = suffix

    def __call__(self, *args, **kwargs) -> str:
        if "term" not in kwargs:
            raise KeyError("No Term provided for surfacing.")
        return self.prefix + kwargs["term"].replace("_", " ") + self.suffix


class TemplateSurfacer(Surfacer):
    """
    A TemplateSurfacer surfaces its prefix, followed by a surfacing of a given Template.
    Specifically, the Template should be given as a kwarg ('result') in __call__() as
    part of a TemplateSequencerResult object. The QAPrompt, chosen_answer_label, and
    Reducer are all used to ensure that the right surface form variation (positive or
    negative; pairing, subsumed, or default) is surfaced. The term_surfacer surfaces
    both source and target Terms in the Template.
    """

    def __init__(self, prefix: str, term_surfacer: Surfacer, forms: SurfaceForms):
        super().__init__(prefix)
        self.term_surfacer = term_surfacer
        self.forms = forms
        self.pos_neg_pattern = re.compile(r"(\[\[(.*?)\|(.*?)]])")

    def __call__(self, instance: Instance, *args, **kwargs) -> str:
        # Grab the ordering.
        if "ordering" not in kwargs:
            raise KeyError("No Ordering provided for surfacing.")
        tree_label, template_id = kwargs["ordering"]
        template = instance.statements[tree_label][template_id]
        reduction_cases = instance.reduction_cases

        # Surface the source and target terms.
        source_term = self.term_surfacer(*args, term=template.source_term, **kwargs)
        target_term = self.term_surfacer(*args, term=template.target_term, **kwargs)

        # If the statement is part of the reduction cases, update the surface form.
        if reduction_cases is not None and template_id in reduction_cases:
            case_link = CaseLink(
                r1_type=instance.pairing.relation_type,
                r2_type=template.relation_type,
                case=reduction_cases[template_id],
            )
            surface_form = self.forms.get(case_link)
            in_reduction = True
        else:
            surface_form = template.surface_form
            in_reduction = False

        # Format the surface form using the surfaced source and target terms.
        text = surface_form.format(source_term, target_term)

        # Find all positive/negative variations in the surface form of the statement.
        match = self.pos_neg_pattern.findall(text)

        # If the statement is part of the reduction cases or is the pairing statement,
        # format according to positive/negative variations.
        if in_reduction or template_id == instance.pairing.id:
            if template_id == instance.pairing.id and not match:
                raise ValueError(
                    "Pairing statement must contain positive/negative variations."
                )

            # Replace pos/neg variations based on label match with the chosen answer.
            is_positive = instance.label == tree_label
            if instance.pairing.flip_negative:
                is_positive = not is_positive
            for group, positive, negative in match:
                text = text.replace(group, positive if is_positive else negative, 1)
        elif match:
            # The statement is neither a pairing statement nor part of the reduction
            # cases, so it CANNOT contain variations.
            raise ValueError(
                "Non-pairing/non-reduction statement cannot contain positive/negative "
                "variations."
            )
        return self.prefix + text


class OrderingSurfacer(Surfacer):
    """
    An OrderingSurfacer surfaces its prefix, followed by a surfacing of all Templates
    in an Instance in the order prescribed by statements_ordering. Each Template in the
    surfaced form is also separated by template_sep.
    """

    def __init__(self, prefix: str, template_separator: str, template_surfacer: Surfacer):
        super().__init__(prefix)
        self.template_separator = template_separator
        self.template_surfacer = template_surfacer

    def __call__(self, instance: Instance, *args, **kwargs) -> str:
        return self.prefix + self.template_separator.join(
            [
                self.template_surfacer(instance, *args, **kwargs, ordering=ordering)
                for ordering in instance.statements_ordering
            ]
        )


class QADataSurfacer(Surfacer):
    """
    A QADataSurfacer surfaces all QA fields in an Instance that are relevant to an LLM
    when it attempts to answer the question. Specifically, it surfaces its prefix,
    followed by the question (unless omit_question is True, in which case the question
    is skipped), followed by the given question_answer_separator, followed by each
    (label, term) pair in the answer_choices (which are individually formatted according
    to the answer_choice_formatter, and then collectively separated using the
    given answer_choice_separator).
    """

    def __init__(
        self,
        prefix: str,
        question_answer_separator: str,
        answer_choice_separator: str,
        answer_choice_formatter: str,
        omit_question: bool,
    ):
        super().__init__(prefix)
        self.question_answer_separator = question_answer_separator
        self.answer_choice_separator = answer_choice_separator
        self.answer_choice_formatter = answer_choice_formatter
        self.omit_question = omit_question

    def __call__(self, instance: Instance, *args, **kwargs) -> str:
        answer_choices = [
            self.answer_choice_formatter.format(label, term)
            for label, term in instance.answer_choices.items()
        ]
        return (
            self.prefix
            + ("" if self.omit_question else instance.question)
            + self.question_answer_separator
            + self.answer_choice_separator.join(answer_choices)
        )


class InstanceSurfacer(Surfacer):
    """
    An InstanceSurfacer surfaces all information in an Instance that is relevant to
    prompting an LLM. Specifically, it surfaces its prefix, then delegates in turn to
    a secondary prefix surfacer (e.g., for instruction prompt text), an ordering
    surfacer (for the Templates of all the Trees), a QA data surfacer (for the question
    and answer choices), and finally a suffix surfacer (e.g., for answer prompt text).
    Each outcome from the delegated surfacers is itself separated by the given
    surfacer_separator. Any delegated surfacer can be None to skip it entirely.
    """

    def __init__(
        self,
        prefix: str,
        surfacer_separator: str,
        prefix_surfacer: Optional[Surfacer],
        ordering_surfacer: Optional[Surfacer],
        qa_data_surfacer: Optional[Surfacer],
        suffix_surfacer: Optional[Surfacer],
    ):
        super().__init__(prefix)
        self.surfacer_separator = surfacer_separator
        self.surfacers = [
            prefix_surfacer,
            ordering_surfacer,
            qa_data_surfacer,
            suffix_surfacer,
        ]

    def __call__(self, instance: Instance, *args, **kwargs) -> str:
        return self.prefix + self.surfacer_separator.join(
            [f(instance, *args, **kwargs) for f in self.surfacers if f is not None],
        )
