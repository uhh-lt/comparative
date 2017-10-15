from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class WordOccurence(BaseEstimator, TransformerMixin):

    def __init__(self, words):
        self.words = words

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feature = []
        for document in documents:
            component = []
            for word in self.words:
                component.append(word.lower() in document.lower())
            feature.append(component)

        reshape = np.array(feature).reshape(-1, len(self.words))
        return reshape

