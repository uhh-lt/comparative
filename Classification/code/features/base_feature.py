from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import numpy as np


class BaseFeature(BaseEstimator, TransformerMixin):
    nlp =  spacy.load('en_core_web_lg')

    def reshape(self, array):
        return np.array(array).reshape(-1, 1)

    def fit(self, X, y):
        return self
