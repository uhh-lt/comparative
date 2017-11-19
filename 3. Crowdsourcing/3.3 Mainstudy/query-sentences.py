import pandas as pd
import requests
import json
from collections import defaultdict

ES_ENDPOINT = "http://localhost:9222/freq-dict/freq"

BASE_URL = 'http://localhost:9222/commoncrawl2/sentence'
SEARCH_URL = BASE_URL + '/_search?size=2000'

QUERY = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "{} AND {}"}}}}]}}}}}}'
QUERY_BETTER = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "({}) AND (\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'

data = pd.read_csv('data/cleaned-brand-list.csv', encoding='latin-1')

markers = set()


def read_jb_json(names):
    for name in names:
        with open(name, 'r') as f:
            data = json.load(f)['results']
            for entry in data[:5]:
                markers.add(entry['key'])


read_jb_json([
    'data/better.json', 'data/worse.json', 'data/superior.json',
    'data/inferior.json'
])
print(' OR '.join(markers))


def query(a, b):
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
limit = 5
pairs = []

for typ in data['type'].unique():
    t_data = data[data['type'] == typ]

    grouped = t_data.groupby(
        'cleaned_name', as_index=False).max().sort_values(
            ['freq'], ascending=False)
    brands = []

    for index, row in grouped.iterrows():
        brands.append((row['cleaned_name'], row['freq'], row['is_entity'], row['source']))

    for current_brand in brands:
        stop = 0
        for next_brand in brands:
            if next_brand != current_brand and used[current_brand] < limit and used[next_brand] < limit and stop <= 50 and current_brand[1] + next_brand[1] > 100 and (current_brand[2] or next_brand[2]) and len(current_brand) >= 3 and len(next_brand) >= 3:
                pairs.append(
                    (current_brand[0], next_brand[0],
                     current_brand[1] + next_brand[1], typ, current_brand[3]))
                stop += 1
                used[next_brand] += 1
                used[current_brand] += 1
    break



obj_s_pairs = []
res = {}
res_ct = 0
for index, pair in enumerate(pairs):
    try:
        a,b, cnt, typ, source = pair
        res[a + '_' + b]= query(a, b)

        for t in res[a + '_' + b][:15]:
            obj_s_pairs.append({
                'id': t['_id'],
                'weight': cnt,
                'a': a,
                'b': b,
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

with open('raw_sentences.json_brand', 'w') as f:
    json.dump({'sentences': res_ct, 'data':res}, f)

with open('sentences_brand.json', 'w') as f:
    json.dump(obj_s_pairs, f)
