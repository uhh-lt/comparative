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
            splitted = nltk.word_tokenize(document)  # TODO: use proper segmenter

            a = splitted.index(self.word_a)
            b = splitted.index(self.word_b)
            if a < b:
                feat.append(' '.join(splitted[splitted.index(self.word_a) + 1:splitted.index(self.word_b)]))
            else:
                feat.append(' '.join(splitted[splitted.index(self.word_b) + 1:splitted.index(self.word_a)]))
        return feat


class ObjectContext(BaseEstimator, TransformerMixin):
    def __init__(self, first):
        self.first = first

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feat = []
        for document in documents:
            splitted = nltk.word_tokenize(document)  # TODO: use proper segmenter

            a = splitted.index('OBJECT_A')
            b = [index for index, value in enumerate(splitted) if value == 'OBJECT_B'][-1]

            if self.first:
                feat.append(' '.join(splitted[:min([a, b])]))
            else:
                feat.append(' '.join(splitted[max([a, b]) + 1:]))
        return feat


class WordPos(BaseEstimator, TransformerMixin):
    def __init__(self, word):
        self.word = word

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feat = []
        for document in documents:
            feat.append([i for i, x in enumerate(nltk.word_tokenize(document)) if x == self.word])

        b = np.zeros([len(feat), len(max(feat, key=lambda x: len(x)))])
        for i, j in enumerate(feat):
            b[i][0:len(j)] = j
        print(len(b))
        return b
