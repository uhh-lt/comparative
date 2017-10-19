from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import nltk

class BeforeAfterWord(BaseEstimator, TransformerMixin):
    def __init__(self, word, before):
        self.word = word
        self.before = 0 if before else 1

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feat = []
        for document in documents:
            splitted = document.lower().split(self.word.lower())

            if len(splitted) <= self.before:
                feat.append('%NONE%')
            else:
                feat.append(document.lower().split(self.word.lower())[self.before])
        return feat


class BetweenWords(BaseEstimator, TransformerMixin):
    def __init__(self, word_a, word_b):
        self.word_a = word_a
        self.word_b = word_b

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feat = []
        for document in documents:
            splitted = nltk.word_tokenize(document.replace)  # TODO: use proper segmenter
            #a = splitted.index(self.word_a)
            if self.word_a not in splitted:
                print(splitted)
            #print(splitted[splitted.index(self.word_a), splitted.index(self.word_b)])
            feat.append(document)
        return feat
