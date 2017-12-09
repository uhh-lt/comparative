from nltk.corpus import wordnet as wn
import nltk
import pandas as pd
import json
import re
import requests
import time
import os
import random

#df = pd.read_csv('data/things.csv')

ES_ENDPOINT = "http://localhost:9222/freq-dict/freq"

freqs = {}

#nltk.download('averaged_perceptron_tagger')


def get_freq(term):
    contains_word = re.compile(r'\w')
    word = str(term).strip().replace("_", " ").encode('utf-8').decode('utf-8')
    if word in freqs.keys():
        print(word, freqs[word])
        return freqs[word]
    else:
        try:
            query = '{{ "query": {{ "query_string": {{ "fields": ["word"], "query": "\\"{}\\"" }} }} }}'.format(
                word)
            res = requests.post(
                url=ES_ENDPOINT + "/_search", data=query.encode('utf-8'))
            hits = res.json()['hits']
            if len(hits) > 0 and len(hits['hits']):
                hits_data = hits['hits'][0]['_source']
                freqs[word] = hits_data['freq']
                return hits_data['freq']
            freqs[word] = 0
            return "0"
        except KeyError as e:
            print(e)
            freqs[word] = 0
            return "0"


def get_antonym(term):
    synset = wn.synsets(term)
    for syn in synset:
        for l in syn.lemmas():
            if l.antonyms():
                for ant in l.antonyms():
                    if ant.synset().pos() is 'n':
                        print(ant.name(), (ant.synset().pos()))
                        return ant.name()

    return None


pairs = []
used = []
p = {}

for synset in wn.all_synsets('n'):
    a = synset.lemma_names()[0]
    b = get_antonym(a)
    if b is not None and a not in used and b not in used:
        p[a] = b
        used.append(a)
        used.append(b)
        freq_a = get_freq(a)
        freq_b = get_freq(b)
        pairs.append({
            'use_marker': random.random() >= 0.1,
            'type': 'wordnet',
            'a': {
                'word': a,
                'freq': freq_a
            },
            'b': {
                'word': b,
                'freq': freq_b
            },
        })
        p[a] = b

folder = time.strftime("%H-%M")
if not os.path.exists(folder):
    os.makedirs(folder)
with open('{}/wordnet-pairs.json'.format(folder), 'w') as f:
    print(len(pairs))
    json.dump(pairs, f)

with open('{}/wordnet-pairs.tsv'.format(folder), 'w') as f:
    f.write('word_a\tfreq_a\tword_b\tfreq_b\ttype\tuse_marker\n')
    for p in pairs:
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(p['a']['word'], p['a']['freq'], p['b']['word'], p['b']['freq'], p['type'],p['use_marker'] ))








#df['antonym'] = df.apply(lambda row: get_antonym(row['noun']), axis=1)

#df.to_csv('foo2.csv')

#df = pd.read_csv('foo.csv')

#df[df['antonym'].notnull()].to_csv('random.csv', index=False)
