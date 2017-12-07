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
parser.add_argument('-n', action='store', dest='look_at_next')
parser.add_argument('-m', action='store', dest='min_freq')
parser.add_argument('-l', action='store', dest='max_appearance')

args = parser.parse_args()
file = args.file
limit = int(args.max_appearance)
next_frequent = int(args.look_at_next)
min_freq = int(args.min_freq)

res = {}
used = defaultdict(int)
pairs = []
data = pd.read_csv('../data/cleaned-{}.csv'.format(file), encoding='latin-1')

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
    
            if next_brand != current_brand and used[a.lower()] < limit and used[b.lower(
            )] < limit and stop <= next_frequent and current_brand[1] >= min_freq and next_brand[1] >= min_freq:
                pairs.append({
                    'type' : current_brand[2],
                    'use_marker' : random.random() >=  0.1,
                    'a': {
                        'word': current_brand[0],
                        'freq': current_brand[1],

                    },
                    'b': {
                        'word': next_brand[0],
                        'freq': next_brand[1],
                    }
                })
                used[a.lower()] +=1
                used[b.lower()] +=1
sort = sorted(pairs, key=lambda x: x['a']['freq'],reverse=True)
with open('pairs-{}.json'.format(file), 'w') as f:
    json.dump({'meta': {'type' : file, 'max_apperance_per_term' : limit, 'next_frequent' : next_frequent, 'min_freq' :min_freq ,  'pairs' : len(pairs)}, 'data':sort}, f)
    print('end')
with open ('pairs-{}.tsv'.format(file), 'w') as f:
    f.write('word_a\tfreq_a\tword_b\tfreq_b\ttype\tuse_marker\n')
    for p in sort:
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(p['a']['word'], p['a']['freq'], p['b']['word'], p['b']['freq'], p['type'],p['use_marker'] ))
