from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import nltk


def all_occ(sentence, word):
    return [i for i, w in enumerate(nltk.word_tokenize(sentence)) if w == word]

class BetweenWords(BaseEstimator, TransformerMixin):

    def __init__(self, word_a, word_b, first_occ_a=True, first_occ_b=True, ignore=False):
        """
        Returns all words between the two given; if b appears before a, all words between b and a
        :param word_a: the word
        :param word_b: the word
        :param first_occ_a: use the first occurence of a; if False, the last one is used
        :param first_occ_b: use the first occurence of b; if False, the last one is used
        """
        self.word_a = word_a
        self.word_b = word_b
        self.first_occ_a = first_occ_a
        self.first_occ_b = first_occ_b
        self.ignore = ignore

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feat = []
        if not self.ignore:
            for document in documents:
                splitted = nltk.word_tokenize(document)

                a_occ = sorted( all_occ(document, self.word_a))
                b_occ = sorted(all_occ(document, self.word_b))
                try:
                    boundary = sorted([a_occ[0] if self.first_occ_a else a_occ[-1], b_occ[0] if self.first_occ_b else b_occ[-1]])
                    feat.append(' '.join(splitted[boundary[0]+1:boundary[1]]))
                except Exception as e:
                    print(document)
        else:
            feat = ["Word"] * len(documents)

        return feat


class BeforeAfterWord(BaseEstimator, TransformerMixin):
    def __init__(self, word_a,before=True,first_occ=True,ignore=False):
        self.word_a = word_a
        self.first_occ = first_occ
        self.before = before
        self.ignore = ignore

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        feat = []
        for document in documents:
                splitted = nltk.word_tokenize(document)
                a_occ = sorted( all_occ(document, self.word_a))
                if self.before:
                    boundary = sorted([0, a_occ[0] if self.first_occ else a_occ[-1]])
                    feat.append(' '.join(splitted[boundary[0]+1:boundary[1]]))
                else:
                    boundary = sorted([a_occ[0] if self.first_occ else a_occ[-1],len(document)])
                    feat.append(' '.join(splitted[boundary[0]:boundary[1]]))

        return feat


