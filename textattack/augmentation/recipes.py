import random

import textattack

from . import Augmenter

DEFAULT_CONSTRAINTS = [
    textattack.constraints.pre_transformation.RepeatModification(),
    textattack.constraints.pre_transformation.StopwordModification(),
]



class EasyDataAugmenter(Augmenter):
    """

        An implementation of Easy Data Augmentation, which combines:

        - WordNet synonym replacement
        - Word deletion
        - Word order swaps
        - Synonym insertion

        in one augmentation method.

        "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" (Wei and Zou, 2019)
        https://arxiv.org/abs/1901.11196

    :param alpha: fraction of words to modify each iteration.
    :param n_aug: how many augmented examples to create from each existing input.

    """

    def __init__(self, alpha, n_aug):
        self.alpha = alpha
        self.transformations_per_example = n_aug
        n_aug_each = max(n_aug // 4, 1)

        self.synonym_replacement = WordNetAugmenter(transformations_per_example=n_aug_each)
        self.random_deletion = DeletionAugmenter(transformations_per_example=n_aug_each)
        self.random_swap = SwapAugmenter(transformations_per_example=n_aug_each)
        self.random_insertion = SynonymInsertionAugmenter(transformations_per_example=n_aug_each)

    def _set_words_to_swap(self, num):
        self.synonym_replacement.num_words_to_swap = num
        self.random_deletion.num_words_to_swap = num
        self.random_swap.num_words_to_swap = num
        self.random_insertion.num_words_to_swap = num

    def augment(self, text):
        attacked_text = textattack.shared.AttackedText(text)
        num_words_to_swap = max(1, int(self.alpha*len(attacked_text.words)))
        self._set_words_to_swap(num_words_to_swap)
        
        augmented_text = [attacked_text.printable_text()]
        augmented_text += self.synonym_replacement.augment(text)
        augmented_text += self.random_deletion.augment(text)
        augmented_text += self.random_swap.augment(text)
        augmented_text += self.random_insertion.augment(text)

        random.shuffle(augmented_text)
        return augmented_text[:self.transformations_per_example]

class SwapAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import RandomSwap
        transformation = RandomSwap()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)

class SynonymInsertionAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import RandomSynonymInsertion
        transformation = RandomSynonymInsertion()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class WordNetAugmenter(Augmenter):
    """ Augments text by replacing with synonyms from the WordNet thesaurus. """

    def __init__(self, **kwargs):
        from textattack.transformations import WordSwapWordNet

        transformation = WordSwapWordNet()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)

class DeletionAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import WordDeletion
        transformation = WordDeletion()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class EmbeddingAugmenter(Augmenter):
    """ Augments text by transforming words with their embeddings. """

    def __init__(self, **kwargs):
        from textattack.transformations import WordSwapEmbedding

        transformation = WordSwapEmbedding(
            max_candidates=50, embedding_type="paragramcf"
        )
        from textattack.constraints.semantics import WordEmbeddingDistance

        constraints = DEFAULT_CONSTRAINTS + [WordEmbeddingDistance(min_cos_sim=0.8)]
        super().__init__(transformation, constraints=constraints, **kwargs)


class CharSwapAugmenter(Augmenter):
    """ Augments words by swapping characters out for other characters. """

    def __init__(self, **kwargs):
        from textattack.transformations import CompositeTransformation
        from textattack.transformations import (
            WordSwapNeighboringCharacterSwap,
            WordSwapRandomCharacterDeletion,
            WordSwapRandomCharacterInsertion,
            WordSwapRandomCharacterSubstitution,
            WordSwapNeighboringCharacterSwap,
        )

        transformation = CompositeTransformation(
            [
                # (1) Swap: Swap two adjacent letters in the word.
                WordSwapNeighboringCharacterSwap(),
                # (2) Substitution: Substitute a letter in the word with a random letter.
                WordSwapRandomCharacterSubstitution(),
                # (3) Deletion: Delete a random letter from the word.
                WordSwapRandomCharacterDeletion(),
                # (4) Insertion: Insert a random letter in the word.
                WordSwapRandomCharacterInsertion(),
            ]
        )
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)
