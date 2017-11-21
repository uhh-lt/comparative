import pandas as pd
import json


df = pd.read_json(path_or_buf='sentences_brand.json',orient='columns')
with open('sample.csv', 'w') as f:
    df.sample(500)[['highlighted']].to_csv(f)