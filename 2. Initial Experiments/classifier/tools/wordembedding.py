from sklearn.base import BaseEstimator, TransformerMixin
import gensim
from collections import defaultdict
from sklearn.feature_extraction.text import  TfidfVectorizer
import numpy as np


class WordEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/"""

    def __init__(self, sentences, size=100):
        self.model = gensim.models.Word2Vec(sentences, size=size)
        self.w2v = dict(zip(self.model.wv.index2word, self.model.wv.syn0))
        self.size = size

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
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                    or [np.zeros(self.size)], axis=0)
            for words in X])
