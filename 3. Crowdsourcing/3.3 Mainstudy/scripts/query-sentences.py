import pandas as pd
import requests
import json
from collections import defaultdict
import time
import os

ES_ENDPOINT = "http://localhost:9222/freq-dict/freq"

BASE_URL = 'http://localhost:9222/commoncrawl2/sentence'
SEARCH_URL = BASE_URL + '/_search?size=15'

QUERY_BETTER = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "({}) AND (\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'
QUERY = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "(\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'

NAME = 'brand-list'

data = pd.read_csv(
    '../data/old/cleaned-{}.csv'.format(NAME), encoding='latin-1')

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


used = defaultdict(int)
limit = 15
next_frequent = 5
pairs = []

res = {}
query_count = 0
obj_s_pairs = []
for typ in data['source'].unique():
    t_data = data[data['source'] == typ]

    grouped = t_data.groupby(
        'cleaned_name', as_index=False).max().sort_values(
            ['freq'], ascending=False)
    brands = []

    for index, row in grouped.iterrows():
        brands.append((row['cleaned_name'], row['freq'], row['source']))

    for current_brand in brands:
        stop = 0
        for next_brand in brands:
            a = current_brand[0]
            b = next_brand[0]
            if next_brand != current_brand and used[a.lower()] < limit and used[b.lower()] < limit and stop < next_frequent  and len(a) >= 3 and len(b) >= 3:
                try:
                    query_result = query(a, b, query_count)
                    res[a + '_' + b] = query_result

                    sentence = query_result[:1]
                    if len(query_result) > 0:
                        pairs.append({
                            'a': {
                                'word': a,
                                'freq': current_brand[1]
                            },
                            'b': {
                                'word': b,
                                'freq': next_brand[1]
                            }
                        })
                    for t in res[a + '_' + b][:1]:
                        obj_s_pairs.append({
                            'id':
                            t['_id'],
                            'weight':
                            current_brand[1] + next_brand[1],
                            'a':
                            a,
                            'b':
                            b,
                            'without-marker':
                            query_count % 10 == 0,
                            'typ':
                            typ,
                            'source':
                            typ,
                            'sentence':
                            t['_source']['text'],
                            'highlighted':
                            t['highlight']['text']
                        })
                    query_count += 1
                    stop += 1
                    print(a, used[a.lower()], b, used[b.lower()], stop,
                          len(res), len(obj_s_pairs))
                    used[a.lower()] += 1 if len(query_result) > 0 else 0
                    used[b.lower()] += 1 if len(query_result) > 0 else 0
                except Exception as e:
                    print(e)

import time

folder = time.strftime("%H-%M")
if not os.path.exists(folder):
    os.makedirs(folder)

with open('{}/raw-sentences-{}.json'.format(folder, NAME), 'w') as f:
    try:
        json.dump({
            'meta': {
                'marker': markers,
                'next': next_frequent,
                'max_appear': limit,
                'sentences': len(pairs)
            },
            'data': res
        }, f)
    except Exception as e:
        print(e)

with open('{}/sentences-{}.json'.format(folder, NAME), 'w') as f:
    json.dump(obj_s_pairs, f)

with open('{}/pairs-{}.json'.format(folder, NAME), 'w') as f:
    json.dump(pairs, f)

with open('{}/meta-{}.json'.format(folder, NAME), 'w') as f:
    json.dump({
        'marker': markers,
        'search_with_next_frequent': next_frequent,
        'maximum_appearance_per_term': limit,
        'number_of_sentences': len(obj_s_pairs)
    }, f)
