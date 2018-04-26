from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize

class WordTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        result = []
        for document in documents:
            result.append(word_tokenize(document))
            print(self, word_tokenize(document))
        return result