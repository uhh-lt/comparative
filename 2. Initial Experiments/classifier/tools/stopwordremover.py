import nltk
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin


class StopwordRemover(BaseEstimator, TransformerMixin):

    def __init__(self, language='english', remove_punctuation=True):

        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.remove_punctuation = remove_punctuation

    def is_punct(self, token):
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        return token.lower() in self.stopwords if self.remove_punctuation else False

    def fit(self, X, y=None):
        return self

    def normalize(self, document):
        return [token for token in document.split() if not self.is_stopword(token) and not self.is_punct(token)]

    def transform(self, documents):
        for document in documents:
            out = ' '.join(self.normalize(document))
            yield out
