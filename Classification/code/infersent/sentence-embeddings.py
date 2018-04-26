import nltk
import pandas as pd
import torch
from bs4 import BeautifulSoup

frame = pd.DataFrame.from_csv(path='../data/train-data.csv')[:10]
frame['sentence'] = frame.apply(
    lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
    axis=1)

nltk.download('punkt')

infersent = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
print("Loaded")
infersent.set_glove_path('glove.840B.300d.txt')
values = list(frame['raw_text'].values)
infersent.build_vocab(values, tokenize=False)

frame['infersent'] = frame.apply(
    lambda row: list(infersent.encode([row['raw_text']], tokenize=True)[0]),
    axis=1)

frame.to_csv('with-infersent.csv')
