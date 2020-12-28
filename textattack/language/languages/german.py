from textattack.language.language_resource_provision import LanguageResourceProvision
from textattack.transformations.word_swaps.word_swap_embedding import WordSwapEmbedding
from textattack.shared import GensimWordEmbedding
from textattack.language.languages.resource_provision import provide_word2vec


class GermanResourceProvision(LanguageResourceProvision):

    # TODO: cache
    @staticmethod
    def provide_word_swap_embedding():
        word2vec_embedding = provide_word2vec('cc.de.300.vec')
        embedding = GensimWordEmbedding(word2vec_embedding)
        return WordSwapEmbedding(embedding=embedding)
