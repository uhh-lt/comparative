import pandas as pd
import requests
import json
from collections import defaultdict
import time
import os

ES_ENDPOINT = "http://localhost:9222/freq-dict/freq"

BASE_URL = 'http://localhost:9222/commoncrawl2/sentence'
SEARCH_URL = BASE_URL + '/_search?size=150'

QUERY_BETTER = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "({}) AND (\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'
QUERY = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "(\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'

NAME = 'wordnet'

with open('../final-data/wordnet/wordnet-pairs.json') as d:
    data = json.load(d)

markers_better = [
    'better', 'easier', 'faster', 'nicer', 'wiser', 'cooler', 'decent',
    'safer', 'superior', 'solid', 'teriffic'
]
markers_worse = [
    'worse', 'harder', 'slower', 'poorly', 'uglier', 'poorer', 'lousy',
    'nastier', 'inferior', 'mediocre'
]

markers = markers_better + markers_worse


def query(a, b, counter):
    if counter % 10 == 0:
        query_string = QUERY.format(a, b)
    else:
        query_string = QUERY_BETTER.format(' OR '.join(markers).strip(), a, b)
    headers = {'Content-Type': 'application/json'}
    res = requests.post(
        SEARCH_URL, data=query_string, headers=headers, timeout=60).json()
    try:
        hits = res['hits']['hits']
        return hits
    except KeyError as e:
        print(e)


limit = 10
pairs = []
res = {}
query_count = 0
obj_s_pairs = []
for pair in data:
    a = pair['a']['word'].replace("_", " ")
    b = pair['b']['word'].replace("_", " ")
    try:
        query_result = query(a, b, query_count)
        res[a + '_' + b] = query_result


        for t in res[a + '_' + b][:limit]:
            obj_s_pairs.append({
                'id':
                t['_id'],
                'weight':
                pair['a']['freq'] + pair['b']['freq'],
                'a':
                a,
                'b':
                b,
                'without-marker':
                query_count % 10 == 0,
                'typ':
                'wordnet',
                'source':
                'wordnet',
                'sentence':
                t['_source']['text'],
                'highlighted':
                t['highlight']['text']
            })
        query_count += 1
        print(a, b, len(res), len(obj_s_pairs))
    except Exception as e:
        print(e)

import time

folder = time.strftime("%H-%M")
if not os.path.exists(folder):
    os.makedirs(folder)

with open('{}/raw-sentences-{}.json'.format(folder,NAME), 'w') as f:
    try:
        json.dump(res, f)
    except Exception as e:
        print(e)

with open('{}/sentences-{}.json'.format(folder, NAME), 'w') as f:
    json.dump(obj_s_pairs, f)

with open('{}/meta-{}.json'.format(folder, NAME), 'w') as f:
    json.dump({
        'marker': markers,
        'maximum_appearance_per_term': limit,
        'number_of_sentences': len(obj_s_pairs)
    }, f)
