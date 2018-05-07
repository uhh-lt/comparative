import json
import numpy as np
import pandas as pd
import spacy

from .base_feature import BaseFeature

nlp = spacy.load('en_core_web_lg')


class SelectDataFrameColumn(BaseFeature):

    def __init__(self, column, value_transform=lambda x: x):
        """Selects one column from a dataframe. Optionally applies a function on the selected column"""
        self.column = column
        self.value_transform = value_transform

    def transform(self, dataframe):
        vectors = dataframe[self.column].values.tolist()
        tolist = [self.value_transform(x) for x in vectors]
        reshape = np.array(tolist)
        return reshape


class PathEmbeddingFeature(BaseFeature):

    def __init__(self, path_file, only_with_path=False):
        """Searches the path embeddings for all sentences in the data frame"""
        self.path_file = path_file
        self.only_with_path = only_with_path

    def transform(self, dataframe):
        paths = pd.read_csv(self.path_file)
        feat = []
        for i, row in dataframe.iterrows():
            embedding = paths[paths.id == i].embedding
            tolist_ = json.loads(embedding.values.tolist()[0])
            feat.append(tolist_)
        assert len(feat) == len(dataframe)
        return feat


class WordVector(BaseFeature):

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        feat = []
        for i, row in df.iterrows():
            a = list(nlp(row['object_a']).sents)[0][0].vector
            b = list(nlp(row['object_b']).sents)[0][0].vector
            feat.append(np.concatenate((a, b)))

        return feat


class POSTransformer(BaseFeature):
    def fit(self, x, y=None):
        return self

    def transform(self, sentence):
        feat = []
        for s in sentence:
            a = list(nlp(s).sents)[0]
            p = ' '.join([t.pos_ for t in a])
            feat.append(p)

        return feat
