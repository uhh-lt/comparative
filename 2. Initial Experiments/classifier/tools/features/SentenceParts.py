from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import nltk


def all_occ(sentence, word):
    return [i for i, w in enumerate(nltk.word_tokenize(sentence)) if w == word]

class BetweenWords(BaseEstimator, TransformerMixin):
    """
        :keyword abc
    """
    def __init__(self, word_a, word_b,min_a=True, min_b=True):
        self.word_a = word_a
        self.word_b = word_b
        self.min_a = min_a
        self.min_b

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feat = []
        for document in documents:
            splitted = nltk.word_tokenize(document)

            a_occ = sorted( all_occ(document, self.word_a))
            b_occ = sorted(all_occ(document, self.word_b))






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
            try:
                a = splitted.index('OBJECT_A')
                b = [index for index, value in enumerate(splitted) if value == 'OBJECT_B'][-1]

                if self.first:
                    feat.append(' '.join(splitted[:min([a, b])]))
                else:
                    feat.append(' '.join(splitted[max([a, b]) + 1:]))
            except Exception as e:
                print(splitted)
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
