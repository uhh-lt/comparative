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
            if len(doc) == 0:
                result.append(np.array([0] * 300))
            else:
                pre = MeanWordEmbedding.nlp(doc)
                result.append(pre.vector)
        return np.array(result).reshape(len(result), -1)

    def get_feature_names(self):
        return ['mwe_{}'.format(i) for i in range(0, 35000)]
