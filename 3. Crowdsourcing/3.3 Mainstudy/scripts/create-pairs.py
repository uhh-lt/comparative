import pandas as pd
import grequests
import json
from collections import defaultdict
import time
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-f', action='store', dest='file')

args = parser.parse_args()
file = args.file

res = {}
used = defaultdict(int)
pairs = []
hashes = set()
data = pd.read_csv('../data/cleaned-{}.csv'.format(file), encoding='latin-1')
cnt = 0
for typ in data['type'].unique():
    t_data = data[data['type'] == typ]

    grouped = t_data.groupby(
        'cleaned_name', as_index=False).max().sort_values(
            ['freq'], ascending=False)
    brands = []

    for index, row in grouped.iterrows():
        brands.append((row['cleaned_name'], row['freq'], row['type']))

    for current_brand in brands:
        stop = 0
        for next_brand in brands:
            a = current_brand[0]
            b = next_brand[0]
            h = hash(a) + hash(b)
            if h not in hashes and next_brand != current_brand:
                hashes.add(h)
                pairs.append({
                    'type': current_brand[2],
                    'use_marker': cnt % 10 != 0,
                    'a': {
                        'word': current_brand[0],
                        'freq': current_brand[1],
                    },
                    'b': {
                        'word': next_brand[0],
                        'freq': next_brand[1],
                    }
                })
                cnt += 2
                used[a.lower()] += 1
                used[b.lower()] += 1
"""
with open('pairs-{}.json'.format(file), 'w') as f:
    json.dump({'meta': {'type' : file, 'max_apperance_per_term' : limit, 'next_frequent' : next_frequent, 'min_freq' :min_freq ,  'pairs' : len(pairs)}, 'data':sort}, f)
"""
with open('pairs-{}.tsv'.format(file), 'w') as f:
    f.write('word_a\tfreq_a\tword_b\tfreq_b\ttype\tuse_marker\n')
    for p in pairs:
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            p['a']['word'], p['a']['freq'], p['b']['word'], p['b']['freq'],
            p['type'], p['use_marker']))
