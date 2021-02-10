"""
Language Models Constraint
---------------------------

"""

from abc import ABC, abstractmethod

from textattack.constraints import Constraint
from math import exp


class LanguageModelConstraint(Constraint, ABC):
    """Determines if two sentences have a swapped word that has a similar
    probability according to a language model.

    Args:
        max_log_prob_diff (float): Maximum tolerated likelihood difference `D` from original `w` to replacement word
             `w'`. Computed as `d := log p(w) - log p(w')`. Attacked text will be rejected if `d < D`. Note that for a
             constant difference between `p(w)` and `p(w')`, the log difference will vary if the asbolute probabilities
             change: `log p(w) - log p(w') =/= log p(w + epsilon) - log p(w' + epsilon)`.
        max_prob_diff (float): Maximum tolerated likelihood difference `D` from original `w` to replacement word `w'`.
            Computed as `d := p(w) - p(w')`. Attacked text will be rejected if `d < D`

        compare_against_original (bool):  If `True`, compare new `x_adv` against the original `x`.
            Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(self, max_log_prob_diff=None, max_prob_diff=None, compare_against_original=True):
        if (max_log_prob_diff is None) == (max_prob_diff is None):
            raise ValueError("Must set either max_log_prob_diff xor max_prob_diff")
        # maps probs to the desired ring, assuming log probs as the domain
        self.ring = (lambda x: x) if max_log_prob_diff is not None else exp
        self.max_diff = max_log_prob_diff if max_log_prob_diff else max_prob_diff
        if self.max_diff < 0:
            raise ValueError("Maximum probability difference has to be greater than or equal to zero")
        super().__init__(compare_against_original)

    @abstractmethod
    def get_log_probs_at_index(self, text_list, word_index):
        """Gets the log-probability of items in `text_list` at index
        `word_index` according to a language model."""
        raise NotImplementedError()

    def _check_constraint(self, transformed_text, reference_text):

        def replacement_word_less_likely_than_reference_word_by_at_least_max_diff(transformed_prob, ref_prob):
            return self.ring(transformed_prob) < self.ring(ref_prob) - self.ring(self.max_diff)

        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply language model constraint without `newly_modified_indices`"
            )

        for i in indices:
            probs = self.get_log_probs_at_index((reference_text, transformed_text), i)
            if len(probs) != 2:
                raise ValueError(
                    f"Error: get_log_probs_at_index returned {len(probs)} values for 2 inputs"
                )
            ref_prob, transformed_prob = probs
            if replacement_word_less_likely_than_reference_word_by_at_least_max_diff(transformed_prob, ref_prob):
                return False

        return True

    def extra_repr_keys(self):
        return ["max_log_prob_diff"] + super().extra_repr_keys()
