
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class POSCount(BaseEstimator, TransformerMixin):
    def __init__(self, spacy, pos_type):
        self.spacy = spacy
        self.pos_type = pos_type

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        all_pos = []
        for document in documents:
            counter = 0
            for token in self.spacy(document):
                if token.pos == self.pos_type:
                    counter += 1
            all_pos.append(counter)

        return np.array(all_pos).reshape(-1,1)

    def get_feature_names(self):
        return [self.pos_type+'_COUNT']


class POSSequence(BaseEstimator, TransformerMixin):

    def __init__(self, spacy):
        self.spacy = spacy

    def fit(self,X,y=None):
        return self

    def transform(self, documents):
        res = []
        for document in documents:
            doc = self.spacy(document)
            seq =' '.join([token.pos_ for token in doc])
            res.append(seq)
            yield seq



class NERCount(BaseEstimator, TransformerMixin):
    def __init__(self, spacy):
        self.spacy = spacy

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        all_count = []
        for document in documents:
            all_count.append(len(self.spacy(document).ents))

        return np.array(all_count).reshape(-1,1)

    def get_feature_names(self):
        return [self.pos_type+'_COUNT']

