import requests
import spacy
import editdistance
from collections import defaultdict
from pprint import pprint
import json
import re
import time

ES_ENDPOINT = 'http://localhost:9222/fd2/freq'
#url_pattern = 'http://ltmaggie.informatik.uni-hamburg.de/jobimviz/ws/api/trigram/jo/similar/{}?numberOfEntries=10&format=json'
url_pattern = 'http://ltmaggie.informatik.uni-hamburg.de/jobimviz/ws/api/stanford/jo/senses/{}%23NP?sensetype=CW&format=json'
groups = []
dups = set()
lemmas = set()
header = 'name,source,type,cleaned_name,freq\n'
pattern = '{},{},{},{},{}\n'
words = []
nlp = spacy.load('en')

def get_words(term):
    req = requests.get(url_pattern.format(term.lower())).json()['result']
    for res in req:
        sense = res['senses']
        added = 0
        for word in sense[:5]:
            term2 = word.split('#')[0]
            token = nlp(term2)
            pos = [t.pos_ for t in token]
            lemmatized = ' '.join([t.lemma_ for t in token])
            h = hash(term.lower()) + hash(term2.lower())
            if lemmatized not in lemmas and h not in dups and editdistance.eval(
                    term, term2) >= 4 and added <= 5:
                if 'NOUN' in pos and 'PUNCT' not in pos and len(lemmatized) > 5:
                    dups.add(h)
                    lemmas.add(lemmatized)
                    print(term, lemmatized, h)
                    freq = get_freq(lemmatized)
                    line = pattern.format(
                        lemmatized, 'jobimtext',
                        'seed=' + term + ';cui=' + res['cui'], lemmatized, freq)
                    if freq > 0:
                        groups.append(line)
                        words.append('{}\t{}\t({})'.format(term, lemmatized, ', '.join(pos)))
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
        if '#' not in line:
            get_words(line.strip())
            time.sleep(1)

with open('../data/cleaned-jbt.csv', 'w') as f:
    f.write(header)
    for line in groups:
        f.write(line)

with open('../data/words-jbt.csv', 'w') as f:
    for line in sorted(words):
        f.write(line+'\n')