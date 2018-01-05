from sklearn.base import BaseEstimator, TransformerMixin
from .base_feature import BaseFeature


class MeanWordEmbedding(BaseFeature):
    """Mean Word Embeddings with the Spacy Standard Model"""

    def transform(self, documents):
        result = []
        for doc in documents:
            pre = MeanWordEmbedding.nlp(doc)
            result.append(pre.vector)
        return result
