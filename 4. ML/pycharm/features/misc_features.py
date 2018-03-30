from .base_feature import BaseFeature
from collections import defaultdict
import numpy as np


class PositionOfObjects(BaseFeature):

    def transform(self, dataframe):
        result = []
        for index, row in dataframe.iterrows():
            pos = row['sentence']
            result.append(sorted([pos.index(row['a']), pos.index(row['b'])]))
        return result


class SelectAllPaths(BaseFeature):

    def __init__(self, path_frame):
        self.path_frame = path_frame

    def transform(self, dataframe):
        result = []
        for index, row in dataframe.iterrows():
            id_ = self.path_frame[self.path_frame['id'] == row['id']]
            paths = id_.path.values.tolist()
            result.append('. '.join(paths))
        return result

class MeanPathVector(BaseFeature):

    def __init__(self,w2v,path_frame):
        self.w2v = w2v
        self.path_frame = path_frame

    def transform(self, dataframe):
        result = []
        for index, row in dataframe.iterrows():
            id_ = self.path_frame[self.path_frame['id'] == row['id']]
            paths = id_.path.values.tolist()
            vec = np.zeros(300)
            for path in paths[:1]:
                s_vec = np.zeros(300)
                tokens = path.split(' ')
                for token in tokens:
                    try:
                        s_vec += self.w2v.wv[token]
                    except KeyError as e:
                        print(e)
                s_vec /= len(token)
                vec += s_vec
            print(vec)
            result.append(vec)



        return result

class PositionOfWord(BaseFeature):

    def __init__(self, word, lowercase=True):
        self.lowercase = lowercase
        self.word = word

    def transform(self, documents):
        result = []
        for doc_ in documents:
            doc = [self.process(t.text.lower()) for t in PositionOfWord.nlp(doc_)]
            try:
                result.append(doc.index(self.process(self.word)))
            except ValueError as e:
                result.append(-1)
        return np.reshape(result, -1, 1)

    def process(self, word):
        return word.lower() if self.lowercase else word
