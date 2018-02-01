import pandas as pd
from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.model_selection import StratifiedKFold


def load_data(file_name, min_confidence=0.67, binary=False, source=None):
    print('### Minimum Confidence {}'.format(min_confidence))
    frame = pd.concat([df.from_csv(path='data/' + file_name),
                       df.from_csv(path='data/do-not-touch/held-out-data.csv')]).drop_duplicates(keep=False,
                                                                                                 subset='id')

    frame = frame[frame['label:confidence'] >= min_confidence]
    frame['raw_text'] = frame.apply(
        lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
        axis=1)
    if binary:
        frame['label'] = frame.apply(lambda row: row['label'] if row['label'] == 'NONE' else 'ARG', axis=1)
    if source is not None:
        frame = frame[frame['type'] == source]
    return frame


def k_folds(splits, data, random_state=42):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits, random_state=random_state)
    for train_index, test_index in k_fold.split(data,
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]
