from sklearn.base import BaseEstimator, TransformerMixin
from .base_feature import BaseFeature
import numpy as np


class MeanWordEmbedding(BaseFeature):
    """Mean Word Embeddings with the Spacy Standard Model"""

    def __init__(self):
        self.length = 0

    def transform(self, documents):
        self.length = len(documents)
        result = []
        for doc in documents:
            mwe = np.array([float(0)] * 384)
            if len(doc) == 0:
                result.append(np.array([float(0)] * 384))
            else:
                pre = MeanWordEmbedding.nlp(doc)
                for token in pre:
                    mwe += token.vector
                mwe /= len(pre)
                result.append(mwe)

        return np.array(result).reshape(len(result), -1)

    def get_feature_names(self):
        return ['mwe_{}'.format(i) for i in range(0, 35000)]
