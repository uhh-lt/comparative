from textacy.extract import ngrams
from .base_feature import BaseFeature
from collections import OrderedDict
import numpy as np
from util.ngram import get_ngrams

empty = 'A<<%%EMPTY%%>>A'


class NGram(BaseFeature):
    """Collects all n-grams and creates a boolean n-gram vector"""

    def __init__(self, n_grams, n=1, min_freq=1):
        self.n = n
        self.min_freq = min_freq
        self.n_grams = n_grams

    def transform(self, documents):
        results = []
        for i, doc in enumerate(documents):
            n_gram_dict = OrderedDict(
                sorted({k: 0
                        for k in self.n_grams}.items()))
            n_grams = get_ngrams(doc, self.n, min_freq=self.min_freq)

            for n_gram in n_grams:
                if n_gram in n_gram_dict:
                    n_gram_dict[n_gram] = 1
                else:
                    print(self.n, 'out of dict {}'.format(n_gram))

            result = list(n_gram_dict.values())
            results.append(result)

        r = np.reshape(np.asarray(results), (len(results), -1))
        return r
