from textacy.extract import ngrams
from .base_feature import BaseFeature
from collections import OrderedDict

class NGram(BaseFeature):
    """Collects all n-grams and creates a boolean n-gram vector"""
    
    def __init__(self, n, docs, min_freq=1):
        self.n = n
        self.min_freq = min_freq
        self.n_grams = self.get_all_ngrams(docs)

    def get_ngrams(self, document):
        return sorted([n.text for n in ngrams(NGram.nlp(document), self.n, filter_stops=False, min_freq=self.min_freq)])


    def get_all_ngrams(self, documents):
        n_grams = set()
        for doc in documents:
            for n_gram in self.get_ngrams(doc):
                n_grams.add(n_gram)
        return sorted(list(n_grams))


    def transform(self, documents):
        results = []
        for i, doc in enumerate(documents):
            n_gram_dict = OrderedDict(sorted({k:0 for k in self.n_grams}.items()))
            n_grams = self.get_ngrams(doc)
            for n_gram in n_grams:
                n_gram_dict[n_gram] = 1

            result = list(n_gram_dict.values())
            results.append(result)
        return results
