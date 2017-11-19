import pandas as pd
import re
import json
import requests
import chardet
from collections import defaultdict


ES_ENDPOINT = "http://localhost:9222/freq-dict/freq"

data = pd.read_csv('data/brand-list.csv', encoding='utf-8')
data['cleaned_name'] = data['brand_name'].str.replace(r'[\[\(].*[\)\]]|\B-.*','')
data['cleaned_name'] = data['cleaned_name'].str.strip()

d = defaultdict(bool)

contains_word = re.compile(r'\w')

def get_freq(term):
    try:
        query = '{{ "query": {{ "query_string": {{ "fields": ["word"], "query": "\\"{}\\"" }} }} }}'.format(
            term.strip().encode('utf-8').decode('utf-8'))
        res = requests.post(
            url=ES_ENDPOINT + "/_search", data=query.encode('utf-8'))
        hits = res.json()['hits']
        if len(hits) > 0 and len(hits['hits']):
            hits_data = hits['hits'][0]['_source']
            #print(term, hits_data['freq'])
            d[term] =  len(list(contains_word.findall(hits_data['entity']))) > 0
            print(hits_data['entity'], d[term])
            return hits_data['freq']
        return 0
    except KeyError as e:
        print(e)
        return -1

def add_is_org(term):
    return d['term']


data['freq']= data.apply(
    lambda row: get_freq(row['cleaned_name']), axis=1)

data['is_entity'] = data.apply(lambda row: d[row['cleaned_name']], axis=1)

data.to_csv('data/cleaned-brand-list.csv', index_label='index')
