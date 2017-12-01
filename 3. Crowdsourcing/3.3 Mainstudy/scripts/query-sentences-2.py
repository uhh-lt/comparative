import pandas as pd
import grequests
import json
from collections import defaultdict
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', action='store', dest='file')
parser.add_argument('-n', action='store', dest='look_at_next')
parser.add_argument('-l', action='store', dest='max_appearance')

ES_ENDPOINT = "http://localhost:9222/freq-dict/freq"

BASE_URL = 'http://localhost:9222/commoncrawl2/sentence'
SEARCH_URL = BASE_URL + '/_search?size=100'

QUERY_BETTER = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "({}) AND (\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'
QUERY = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "(\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'

args = parser.parse_args()
NAME = args.file
limit = int(args.max_appearance)
next_frequent = int(args.look_at_next)

print(NAME)
data = pd.read_csv(
    '../data/cleaned-{}.csv'.format(NAME), encoding='latin-1')

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
    if counter % 100 == 0:
        query_string = QUERY.format(a, b)
    else:
        query_string = QUERY_BETTER.format(' OR '.join(markers).strip(), a, b)
    headers = {'Content-Type': 'application/json'}
    req = grequests.post(
        SEARCH_URL, data=query_string, headers=headers, timeout=60)
    res = grequests.map([req])
    try:
        hits = res[0].json()['hits']['hits']
        return hits
    except KeyError as e:
        print('1',e)


used = defaultdict(int)

pairs = []

res = {}
query_count = 0
obj_s_pairs = []
used_ids = []
for typ in data['source'].unique():
    print(typ)
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
            if next_brand != current_brand and used[a.lower()] < limit and used[b.lower()] < limit and stop <= next_frequent  and len(a) >= 3 and len(b) >= 3:
                try:
                    query_result = query(a, b, query_count)
                    res[a + '_' + b] = query_result

                    sentence = query_result[:1]
                    if  len(query_result) > 0:
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
                    for t in res[a + '_' + b]:
                        if t['_id'] not in used_ids and used[a.lower()] < limit and used[b.lower()] < limit:
                            used_ids.append(t['_id'])
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
                                query_count % 100 == 0,
                                'typ':
                                typ,
                                'source':
                                typ,
                                'sentence':
                                t['_source']['text'],
                                'highlighted':
                                t['highlight']['text']
                            })
                            used[a.lower()] += 1
                            used[b.lower()] += 1

                    query_count += 1
                    stop += 1
                    print(a, used[a.lower()], b, used[b.lower()], len(obj_s_pairs))

                except Exception as e:
                    print(e)


import time

folder = time.strftime("%H-%M") + '_next_' + str(next_frequent) + '_max_'+str(limit)
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
        print('3',e)

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




def load(file):
    
    sentences = []
    already = set()
    
    with open(file, 'r') as f:
        data = json.load(f)
        print(len(data))
        for item in data:
            idx, a,b, sentence, with_marker = ((item['id'], item['a'], item['b'], item['sentence'], item['without-marker']))
            if sentence.lower() not in already and is_valid([a, b], sentence):
                sentences.append('{}\t{}\t{}\t{}\t{}'.format(idx,sentence,a,b, with_marker))
                
                already.add(sentence.lower())
    return sentences


def is_valid(words, sentence):
    count = 0
    try:
        for word in words:
            count += sentence.lower().count(word.lower())
        return count == 2
    except Exception as e:
        return False

p = '{}/sentences-{}.json'.format(folder,NAME)
res = load(p)

with open ('{}/sentences-only-{}.tsv'.format(folder,NAME),'w') as f:
    f.write('id\tsentence\ta\tb\twithout_marker\n')
    for sentence in res:
        f.write(sentence+'\n')

