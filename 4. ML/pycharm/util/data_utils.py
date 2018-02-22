import collections

import pandas as pd
from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

CUE_WORDS_WORSE = ["worse", "harder", "slower", "poorly", "uglier", "poorer", "lousy", "nastier", "inferior",
                   "mediocre"]
CUE_WORDS_BETTER = ["better", "easier", "faster", "nicer", "wiser", "cooler", "decent", "safer", "superior", "solid",
                    "teriffic"]


def load_data(file_name, min_confidence=0.67, binary=False, source=None):
    print('### Minimum Confidence {}'.format(min_confidence))
    frame = df.from_csv(path='data/' + file_name)
    frame = frame[frame['label:confidence'] >= min_confidence]
    frame['raw_text'] = frame.apply(
        lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
        axis=1)
    if binary:
        frame['label'] = frame.apply(lambda row: row['label'] if row['label'] == 'NONE' else 'ARG', axis=1)
    if source is not None:
        frame = frame[frame['type'] == source]
    print('Loaded {} training examples'.format(len(frame)))
    print(frame['label'].value_counts())
    return shuffle(frame)


def k_folds(splits, data, random_state=1337):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits, random_state=random_state)
    for train_index, test_index in k_fold.split(data,
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def get_misclassified(predictions, test):
    for idx, row in enumerate(test.iterrows()):
        gold = row[1]['label']
        if predictions[idx] != gold:
            print('{}\tGold: {}\t Pred: {}'.format(row[1]['raw_text'], gold, predictions[idx]))
