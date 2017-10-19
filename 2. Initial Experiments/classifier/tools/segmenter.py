from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from nltk.tokenize import ToktokTokenizer


class Segmenter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.segmenter = ToktokTokenizer()

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feat = []
        for document in documents:
            seg = self.segmenter.tokenize(document)

            feat.append(str(seg))
        print(np.array(feat).reshape((-1,1)))
        return np.array(feat).reshape((-1,1))
