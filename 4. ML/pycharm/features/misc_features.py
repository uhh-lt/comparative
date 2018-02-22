from .base_feature import BaseFeature
from collections import defaultdict
import numpy as np


class PositionOfObjects(BaseFeature):

    def transform(self, dataframe):
        result = []
        for index, row in dataframe.iterrows():
            pos = row['raw_text']
            result.append(sorted([pos.index(row['a']), pos.index(row['b'])]))
        return result


class PositionOfWord(BaseFeature):

    def __init__(self, word, lowercase=False):
        self.lowercase = lowercase
        self.word = word

    def transform(self, documents):
        result = []
        for doc_ in documents:
            doc = [self.process(t.text) for t in PositionOfWord.nlp(doc_)]
            result.append(doc.index(self.process(self.word)))
        return np.reshape(result, -1, 1)

    def process(self, word):
        return word.lower() if self.lowercase else word
