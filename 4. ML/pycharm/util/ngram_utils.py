import spacy
from textacy.extract import ngrams
import json
import datetime

nlp = spacy.load('en')


def get_ngrams(document, n, min_freq=1, filter_punct=True):
    res = sorted([n.text for n in
                      ngrams(nlp(document), n, filter_stops=False, min_freq=min_freq, filter_punct=filter_punct)])
    return res


def get_all_ngrams(documents, n=1, min_freq=1, filter_punct=True):

    start = datetime.datetime.now()
    print('Begin: {}:{}:{}'.format(start.hour, start.minute, start.second))
    n_grams = set()
    for doc in documents:
        for n_gram in get_ngrams(doc, n, min_freq, filter_punct):
            n_grams.add(n_gram)
    start = datetime.datetime.now()
    print('End: {}:{}:{}'.format(start.hour, start.minute, start.second))
    return sorted(list(n_grams))


def read_precomputed(file_name):
    print('Reading ' + file_name)
    with open('data/' + file_name, 'r') as f:
        return json.load(f)
