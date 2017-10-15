from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BeforeFirstObject(BaseEstimator, TransformerMixin):

    def __init__(self, first_object):
        self.first_object = first_object

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield document.lower().split(self.first_object.lower())[0]


