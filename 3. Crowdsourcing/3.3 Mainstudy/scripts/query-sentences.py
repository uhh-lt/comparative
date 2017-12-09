import pandas as pd
import grequests
import json
from collections import defaultdict
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', action='store', dest='file')


ES_ENDPOINT = "http://localhost:9222/fq2/freq"

BASE_URL = 'http://localhost:9222/commoncrawl2/sentence'
SEARCH_URL = BASE_URL + '/_search?size=1000'

QUERY_BETTER = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "({}) AND (\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'
QUERY = ' {{"query" : {{"bool": {{"must": [{{"query_string": {{"default_field" : "text","query" : "(\\"{}\\" AND \\"{}\\")"}}}}]}}}}, "highlight" : {{"fields" : {{"text" : {{}}}} }} }}'

args = parser.parse_args()
NAME = args.file


print(NAME)
data = pd.read_csv(
    '../final-data/pairs/pairs-{}.tsv'.format(NAME),
    sep='\t',
    header=0,
    encoding='latin-1')

markers_better = [
    'better', 'easier', 'faster', 'nicer', 'wiser', 'cooler', 'decent',
    'safer', 'superior', 'solid', 'teriffic'
]
markers_worse = [
    'worse', 'harder', 'slower', 'poorly', 'uglier', 'poorer', 'lousy',
    'nastier', 'inferior', 'mediocre'
]

markers = markers_better + markers_worse


def query(a, b, use_marker):
    if not use_marker:
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


def is_valid(words, sentence):
    count = 0
    try:
        for word in words:
            count += sentence.lower().count(word.lower())
        return count == 2
    except Exception as e:
        return False


used = defaultdict(int)
hits_counter = defaultdict(int)
pairs = []
all_sentences = set()
res = {}
query_count = 0
obj_s_pairs = []
used_ids = []
for typ in list(data['type'].unique()):
    t_data = data[data['type'] == typ]
    for row in list(t_data.iterrows()):
        d = row[1]
        a = d['word_a']
        b = d['word_b']
        if a != b:
            try:
                query_result = query(a, b, d['use_marker'])
                already = set()
                sentences = []
                for hit in query_result:
                    sentence = hit['_source']['text']
                    if sentence.lower() not in already and is_valid([a, b],
                                                                    sentence):
                        sentences.append(hit)
                        already.add(sentence.lower())
                res[a + '_' + b] = sentences
                if len(sentences) >= 100:
                    for t in res[a + '_' + b]:
                        if t['_id'] not in used_ids:
                            all_sentences.add('{}\t{}\t{}\t{}\t{}'.format(t['_id'],t['_source']['text'],a,b,d['use_marker']))
                            used_ids.append(t['_id'])
                            hits_counter[a+'_'+b] += 1
                            obj_s_pairs.append({
                                'id':
                                t['_id'],
                                'a':
                                a,
                                'b':
                                b,
                                'marker':
                                d['use_marker'],
                                'typ':
                                typ,
                                'sentence':
                                t['_source']['text'],
                                'highlighted':
                                t['highlight']['text']
                            })
                            used[a.lower()] += 1
                            used[b.lower()] += 1
                    print('{} {}  {}'.format(a,b,len(sentences)))
                else:
                    hits_counter[a+'_'+b] += int(len(sentences))
                    print('NOT {} {}  {}'.format(a,b,len(sentences)))
                data.set_value(row[0], 'd_type',NAME)
                data.set_value(row[0], 'count',int(hits_counter[a+'_'+b]))
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
                'sentences': len(res)
            },
            'data': res
        }, f)
    except Exception as e:
        print('3',e)

with open('{}/sentences-{}.json'.format(folder, NAME), 'w') as f:
    json.dump(obj_s_pairs, f)


with open('{}/meta-{}.json'.format(folder, NAME), 'w') as f:
    json.dump({
        'marker': markers,
        'number_of_sentences': len(all_sentences)
    }, f)






p = '{}/sentences-{}.json'.format(folder,NAME)

with open ('{}/sentences-only-{}.tsv'.format(folder,NAME),'w') as f:
    f.write('id\tsentence\ta\tb\tmarker\n')
    for sentence in all_sentences:
        f.write(sentence + '\n')

data.to_csv('{}/{}-with-counts.csv'.format(folder,NAME))

