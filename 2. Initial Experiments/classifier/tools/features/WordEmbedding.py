from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer


class GensimW2V(BaseEstimator, TransformerMixin):
    """
    http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
    """

    def __init__(self, file='data/GoogleNews-vectors-negative300.bin'):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(file,unicode_errors='ignore', binary=True)
        self.word2weight = None
        self.dim = 300

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def get_feature_names(self):
        return [self.word2vec]
