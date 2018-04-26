from features.base_feature import BaseFeature
from textacy.extract import ngrams


class NGramTransformer(BaseFeature):
    """
    Expects strings as input, return list list of n-gram lists
    """

    def __init__(self, n=1, min_freq=1, filter_punct=True):
        self.n = n
        self.min_freq = min_freq
        self.filter_punct = filter_punct

    def transform(self, documents):
        result = []
        for doc_ in documents:
            doc = NGramTransformer.nlp(doc_)
            result.append(
                sorted([t.text for t in ngrams(doc, n=self.n, filter_stops=False, min_freq=self.min_freq,
                                               filter_punct=self.filter_punct)]))
        return result

    def fit(self, X, y):
        return self
