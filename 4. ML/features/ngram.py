from textacy.extract import ngrams
from .base_feature import BaseFeature

class NGram(BaseFeature):

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
        for doc in documents:
            result = []
            n_grams = self.get_ngrams(doc)
            for n_gram in self.n_grams:
                if n_gram in self.get_ngrams(doc):
                    result.append(1)
                else:
                    result.append(0)
            results.append(result)


        return results
