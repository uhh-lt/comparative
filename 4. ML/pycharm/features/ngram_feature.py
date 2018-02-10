from .base_feature import BaseFeature
from collections import OrderedDict
import numpy as np
from util.ngram import get_ngrams
from textacy.extract import ngrams as ngram_extract


class NGramFeature(BaseFeature):
    """Collects all n-grams and creates a boolean n-gram vector"""

    def __init__(self, base_n_grams, n=1, with_oov=False):
        self.base_n_grams = base_n_grams
        self.with_oov = with_oov
        self.n = n

    def transform(self, n_gram_lists):
        results = []
        ngram_dict_prototype = self.get_n_gram_dict()

        for ngrams in n_gram_lists:
            ngram_dict = ngram_dict_prototype.copy()
            for n_gram in ngrams:
                if n_gram in ngram_dict:
                    ngram_dict[n_gram] += 1
                elif self.with_oov:
                    ngram_dict['OUT_OF_VOC'] += 1
            results.append(list(ngram_dict.values()))
        r = np.reshape(np.asarray(results), (len(results), -1))
        return r

    def fit(self, X, y):
        return self

    def get_n_gram_dict(self):
        n_grams = self.base_n_grams
        if self.with_oov:
            n_grams += ['OUT_OF_VOC']
        unordered_dict = {k: 0 for k in n_grams}
        items = unordered_dict.items()
        ordered_dict = OrderedDict(sorted(items))
        return ordered_dict

    def get_feature_names(self):
        return ['{}-gram_{}'.format(self.n, n) for n in self.base_n_grams]
