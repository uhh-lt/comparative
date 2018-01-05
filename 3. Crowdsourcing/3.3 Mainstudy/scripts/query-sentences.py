import pandas as pd
import grequests
import json
from collections import defaultdict
import time
import os
import argparse
import random
from textacy.similarity import jaccard
parser = argparse.ArgumentParser()
parser.add_argument('-f', action='store', dest='file')


ES_ENDPOINT = "http://localhost:9222/fq2/freq"

BASE_URL = 'http://localhost:9222/commoncrawl2/sentence'
SEARCH_URL = BASE_URL + '/_search?size=50'

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


def replace_objects(sentence, objects, color=True):
    a1 = '<span style="color: #9A14B2; font-weight: bold">' if color else '*'
    a2 = ':[OBJECT_A]</span>' if color else '*'
    b1 = '<span style="color: #6CB219; font-weight: bold">' if color else '§'
    b2 = ':[OBJECT_B]</span>' if color else '$'
    try:
        sentence_lower = sentence.lower()
        a_start = sentence_lower.index(objects[0].lower())
        b_start = sentence_lower.index(objects[1].lower())
        if a_start < b_start:
            first = sentence[:a_start] + a1 + objects[0] + a2
            middle = sentence[a_start + len(objects[0]):b_start]
            end = b1 + objects[1] + b2 + sentence[b_start + len(objects[1]):]
            return True, first + middle + end
        else:
            first = sentence[:b_start] + a1 + objects[1] + a2
            middle = sentence[b_start + len(objects[1]):a_start]
            end = b1 + objects[0] + b2 + sentence[a_start + len(objects[0]):]
            return True, first + middle + end
    except Exception as e:
        return False, ''


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
        return res[0].json()['hits']['total'], hits
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
all_sentences = list()
res = {}
query_count = 0
obj_s_pairs = []
used_ids = []
for typ in list(data['type'].unique()):
    t_data = data[data['type'] == typ]
    for row in t_data.iterrows():
        d = row[1]
        a = d['word_a'].replace('_','')
        b = d['word_b'].replace('_','')
        if a != b:
            try:
                total, query_result = query(a, b, d['use_marker'])
                already = set()
                sentences = []
                for hit in query_result:
                    sentence = hit['_source']['text']
                    if sentence.lower() not in already and is_valid([a, b],
                                                                    sentence):
                        sentences.append(hit)
                        already.add(sentence.lower())
                res[a + '_' + b] = sentences
                if int(total) >= 50:
                    for t in res[a + '_' + b]:
                        ok, replaced = replace_objects(t['_source']['text'],
                                                       (a, b))
                        ok, replaced2 = replace_objects(
                            t['_source']['text'], (a, b), color=False)
                        if t['_id'] not in used_ids and ok:
                            all_sentences.append({'id' : t['_id'],
                            'raw_text' : t['_source']['text'],
                            'text_readable' : replaced2,
                            'text_html' : replaced,
                            'object_a' : a,
                            'object_b' : b,
                            'marker' : d['use_marker']
                              })

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
                    print('{} {}  {}'.format(a,b,total))
                else:
                    hits_counter[a+'_'+b] += int(len(sentences))
                    #print('NOT {} {}  {}'.format(a,b,len(sentences)))
                data.set_value(row[0], 'd_type',NAME)
                data.set_value(row[0], 'count',int(total))
            except Exception as e:
                print(a,b,e)


selected_sentences = []

for sentence in all_sentences:
    max_sim = 0
    sentence_a = sentence['raw_text']
    for sentence_b in selected_sentences:
        sim = jaccard(sentence_a,sentence_b['raw_text'])
        if sim > max_sim:
            max_sim = sim
    if max_sim < 0.9:
        selected_sentences.append(sentence)

folder = '{}-{}'.format(time.strftime("%H-%M"),NAME)
if not os.path.exists(folder):
    os.makedirs(folder)

with open('{}/raw-sentences-{}.json'.format(folder, NAME), 'w') as f:
    try:
        json.dump({
            'meta': {
                'marker': markers,
                'sentences': len(res)
            },
     'ddata': res
        }, f)
    except Exception as e:
        print('3',e)

#data.to_csv('{}/{}-with-counts.csv'.format(folder,NAME))
#with open('{}/sentences-{}.json'.format(folder, NAME), 'w') as f:
#    json.dump(obj_s_pairs, f)


with open('{}/meta-{}.json'.format(folder, NAME), 'w') as f:
    json.dump({
        'marker': markers,
        'number_of_sentences': len(selected_sentences)
    }, f)

with open('{}/sentences-all-{}.tsv'.format(folder, NAME), 'w') as f:
    f.write('id\ttext_html\ttext_readable\ta\tb\tmarker\traw_text\n')
    for u in selected_sentences:
        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(u['id'], u[
            'text_html'], u['text_readable'], u['object_a'], u['object_b'], u[
                'marker'], u['raw_text']))

with open ('{}/sentences-sample-{}.tsv'.format(folder,NAME),'w') as f:
    f.write('id\ttext_html\ttext_readable\ta\tb\tmarker\traw_text\n')
    for u in random.sample(selected_sentences,2600 ):
        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(u['id'], u['text_html'], u['text_readable'], u['object_a'], u['object_b'], u['marker'], u['raw_text']))
#data.to_csv('{}/{}-with-counts.csv'.format(folder,NAME))
