import spacy
from spacy import displacy
from pathlib import Path
import pandas as pd
import string

df = pd.read_csv('../Classification/all_data_files/experiments/data.csv')

nlp = spacy.load('en')

for i,row in df.iterrows():
    s = row.sentence
    c = row.most_frequent_label
    exclude = set(string.punctuation)
    fn = ''.join(ch for ch in s if ch not in exclude)[:50]

    doc = nlp(s)
    svg = displacy.render(doc, options={'compact' : True}, style='dep')
    output_p = Path('./dependency_parses/{}/{}.svg'.format(c,fn))
    output_p.open('w', encoding='utf-8').write(svg)
