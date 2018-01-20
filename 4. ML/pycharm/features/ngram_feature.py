from .base_feature import BaseFeature
from collections import OrderedDict
import numpy as np
from util.ngram import get_ngrams
from textacy.extract import ngrams as ngram_extract


class NGramFeature(BaseFeature):
    """Collects all n-grams and creates a boolean n-gram vector"""

    def __init__(self, base_n_grams, with_oov=False):
        self.n_grams = base_n_grams
        self.with_oov = with_oov

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
        n_grams = self.n_grams
        if self.with_oov:
            n_grams += ['OUT_OF_VOC']
        return OrderedDict(sorted({k: 0 for k in n_grams}.items()))
