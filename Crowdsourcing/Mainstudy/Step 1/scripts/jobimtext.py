import requests
import spacy
import editdistance
from collections import defaultdict
from pprint import pprint
import json
import re
import time

ES_ENDPOINT = 'http://localhost:9222/fd2/freq'
url_pattern = 'http://ltmaggie.informatik.uni-hamburg.de/jobimviz/ws/api/stanford/jo/similar/{}?numberOfEntries=10&format=json'
#url_pattern = 'http://ltmaggie.informatik.uni-hamburg.de/jobimviz/ws/api/stanford/jo/senses/{}?sensetype=CW&format=json'
groups = []
dups = set()
lemmas = set()
header = 'name,source,type,cleaned_name,freq\n'
pattern = '{},{},{},{},{}\n'
words = []
nlp = spacy.load('en')


def get_words(term):
    uri = url_pattern.format(term)
    term = term.split('%')[0]
    req = requests.get(uri).json()['results']
    for res in req:
        word = res['key']
        added = 0
        term2, pos = word.split('#')
        h = hash(term) + hash(term2)
        if h not in dups and editdistance.eval(term, term2) >= 3:
            if pos.strip() == 'NN' or pos.strip() == 'NP':
                dups.add(h)
                print(term, term2)
                freq = get_freq(term2)
                line = pattern.format(
                    term2, 'jobimtext',
                    'seed=' + term, term2,
                    freq)
                if freq > 0:
                    groups.append(line)
                    words.append('{}\t{}\t({})'.format(
                        term, term2, ', '.join(pos)))
                    added += 1


def get_freq(term):
    contains_word = re.compile(r'\w')
    try:
        query = '{{ "query": {{ "query_string": {{ "fields": ["word"], "query": "\\"{}\\"" }} }} ,  "sort" : {{"freq" : "desc"}} }}'.format(
            str(term).strip().encode('utf-8').decode('utf-8'))
        res = requests.post(
            url=ES_ENDPOINT + "/_search", data=query.encode('utf-8'))
        hits = res.json()['hits']
        if len(hits) > 0 and len(hits['hits']):
            hits_data = hits['hits'][0]['_source']
            return hits_data['freq']
        return 0
    except KeyError as e:
        print(e)
        return -1


with open('../data/jbt-seed-2.txt', 'r') as rIn:
    for line in rIn:
        get_words(line.strip())
        #time.sleep(1)

with open('../data/cleaned-jbt.csv', 'w') as f:
    f.write(header)
    for line in groups:
        f.write(line)

with open('../data/words-jbt.csv', 'w') as f:
    for line in sorted(words):
        f.write(line + '\n')
