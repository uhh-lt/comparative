import pandas as pd
import requests
import json
from collections import defaultdict
import time

ES_ENDPOINT = "http://localhost:9222/freq-dict/freq"

BASE_URL = 'http://localhost:9222/commoncrawl2/sentence'
SEARCH_URL = BASE_URL + '/_search?size=150'

QUERY_BETTER = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "({}) AND (\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'
QUERY = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "(\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'

NAME = 'compsci'

data = pd.read_csv('data/cleaned-{}.csv'.format(NAME), encoding='latin-1')

markers = set()


def read_jb_json(names):
    for name in names:
        with open(name, 'r') as f:
            data = json.load(f)['results']
            for entry in data[:25]:
                markers.add(entry['key'])


read_jb_json([
    'data/marker/better.json', 'data/marker/worse.json',
    'data/marker/superior.json', 'data/marker/inferior.json'
])

#markers = ['better', 'worse']


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
limit = 10
pairs = []
"""
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
            if next_brand != current_brand and used[current_brand] < limit and used[next_brand] < limit and stop <= 50 and current_brand[1] + next_brand[1] > 100 and len(
                    current_brand[0]) >= 3 and len(next_brand[0]) >= 3:
                pairs.append((current_brand[0], next_brand[0],
                              current_brand[1] + next_brand[1], typ, next_brand[2]))
                stop += 1
                used[next_brand] += 1
                used[current_brand] += 1
"""


#for i, row in pd.read_csv('random.csv').iterrows():
#    pairs.append((row['noun'], row['antonym'], -1, 'random', 'internet' ))

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
            if next_brand != current_brand and used[current_brand] < limit and used[next_brand] < limit and stop <= 50 and current_brand[1] + next_brand[1] > 100 and len(
                    a) >= 3 and len(b) >= 3:
                try:
                    if query_count % 1000 == 0:
                        time.sleep(5)
                    query_result = query(a, b, query_count)
                    res[a + '_' + b] = query_result

                    sentence = query_result[:1]
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
                            'with-marker':
                            query_count % 10 == 0,
                            'typ':
                            typ,
                            'source':
                            data['source'],
                            'sentence':
                            t['_source']['text'],
                            'highlighted':
                            t['highlight']['text']
                        })
                    query_count += 1
                    stop += 1
                    print(a, used[current_brand],b,used[next_brand],stop)
                    used[next_brand] += 1 if len(query_result) > 1 else 0
                    used[current_brand] += 1 if len(query_result) > 1 else 0
                except Exception as e:
                    print(e)



"""
for index, pair in enumerate(pairs):
    try:
        a, b, cnt, typ, source = pair
        res[a + '_' + b] = query(a, b, res_ct)

        for t in res[a + '_' + b][:1]:
            obj_s_pairs.append({
                'id': t['_id'],
                'weight': cnt,
                'a': a,
                'b': b,
                'with-marker': res_ct % 10 == 0,
                'typ': typ,
                'source': source,
                'sentence': t['_source']['text'],
                'highlighted': t['highlight']['text']
            })

        res_ct += len(a + '_' + b)
        print(a, b, len(res[a + '_' + b]))
    except Exception as e:
        print(e)

print(len(pairs))
"""
with open('raw-sentences-{}.json'.format(NAME), 'w') as f:
    json.dump({'sentences': res_ct, 'data':res}, f)

with open('sentences-{}.json'.format(NAME), 'w') as f:
    json.dump(obj_s_pairs, f)
