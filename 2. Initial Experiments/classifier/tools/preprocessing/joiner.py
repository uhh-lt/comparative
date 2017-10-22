from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize

class WordJoiner(BaseEstimator, TransformerMixin):

    def __init__(self, join_on=' '):
        self.join_on = join_on

    def fit(self, X, y=None):
        return self

    def transform(self, word_lists):
        result = []
        for word_list in word_lists:
            result.append(self.join_on.join(word_list))
            print(self, self.join_on.join(word_list))
        return result