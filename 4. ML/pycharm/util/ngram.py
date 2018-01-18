import spacy
from textacy.extract import ngrams

nlp = spacy.load('en')


def get_ngrams(document, n, min_freq=1, filter_punct=True):
    return sorted([
        n.text
        for n in ngrams(
            nlp(document),
            n,
            filter_stops=False,
            min_freq=min_freq,filter_punct=filter_punct)
    ])


def get_all_ngrams(documents, n=1, min_freq=1,  filter_punct=True):
    n_grams = set()
    for doc in documents:
        for n_gram in get_ngrams(doc, n, min_freq, filter_punct):
            n_grams.add(n_gram)
    return sorted(list(n_grams))
