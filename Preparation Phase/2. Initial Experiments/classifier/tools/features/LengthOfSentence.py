from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import random

class LengthAnalyzer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        return np.array([len(document) for document in documents]).reshape(-1, 1)


    def get_feature_names(self):
        return ['SENTENCE_LENGTH']


