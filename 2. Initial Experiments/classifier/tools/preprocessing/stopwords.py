from sklearn.base import BaseEstimator, TransformerMixin


class StopwordRemoval(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        result = []
        for document in documents:
            pass
        return result