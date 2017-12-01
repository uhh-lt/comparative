import pandas as pd
import re
import json
import requests
import chardet
from collections import defaultdict


ES_ENDPOINT = "http://localhost:9222/freq-dict/freq"


def clean(col_name, path, file):
    data = pd.read_csv(path + '/' + file, encoding='utf-8')
    data['cleaned_name'] = data[col_name].str.replace(r'[\[\(].*[\)\]]|\B-.*',
                                                      '')
    data['cleaned_name'] = data['cleaned_name'].str.strip()

    data['freq'] = data.apply(
        lambda row: get_freq(row['cleaned_name']), axis=1)

    data.to_csv('../data/cleaned-{}'.format(file), index_label='index')


def get_freq(term):
    contains_word = re.compile(r'\w')
    try:
        query = '{{ "query": {{ "query_string": {{ "fields": ["word"], "query": "\\"{}\\"" }} }} }}'.format(
            str(term).strip().encode('utf-8').decode('utf-8'))
        res = requests.post(
            url=ES_ENDPOINT + "/_search", data=query.encode('utf-8'))
        hits = res.json()['hits']
        if len(hits) > 0 and len(hits['hits']):
            hits_data = hits['hits'][0]['_source']
            print(term, hits_data['freq'])
            return hits_data['freq']
        return 0
    except KeyError as e:
        print(e)
        return -1


clean('name', '../data', 'compsci.csv')
clean('name', '../data', 'brands.csv')
