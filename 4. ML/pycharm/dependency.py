from util.data_utils import load_data
import spacy

nlp = spacy.load('en')

_data = load_data('../data.csv', min_confidence=0, binary=False)[:100]

for i, row in _data.iterrows():
    sentence = row['raw_text']
    print(sentence)