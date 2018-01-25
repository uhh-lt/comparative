import requests
import pandas as pd
from bs4 import BeautifulSoup

frame = pd.DataFrame.from_csv(path='data/train-data.csv')
frame['raw_text'] = frame.apply(
    lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
    axis=1)

sents = {}

for i, row in frame.iterrows():
    print(i)
    sent = requests.get('https://api.sentity.io/v1/sentiment',
                        params={'text': row['raw_text'], 'api_key': '058b2b74f091718a0efc95ac115581719993468b'}).json()

    frame.loc[i, 'pos_sent'] = sent['pos']
    frame.loc[i, 'neg_sent'] = sent['neg']
    sents[i] = sent

frame.to_csv('train-data-with-sent.csv')