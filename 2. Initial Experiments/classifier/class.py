import re
import sys
# sys.path.insert(0, '../')
import nltk
import numpy as np
from pandas import DataFrame
import scipy
from tools import StopwordRemover, LengthAnalyzer, WordOccurence, BeforeFirstObject
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix


def build_data_frame(file_name):
    rows = []
    with open(file_name, 'r') as data:
        for line in data:
            text, label = re.split(r'\t', line)
            rows.append({'text': text.strip(), 'label': label.strip()})
    data_frame = DataFrame(rows)
    return data_frame


# nltk.download('stopwords')

DATA = build_data_frame('data/data.txt')


def run_pipeline(model):
    train, test = train_test_split(DATA, test_size=0.2)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('meta', Pipeline([
                ('length', LengthAnalyzer()),
            ])),
            ('word-occ', Pipeline([
                ('word-occ', WordOccurence(["better", "worse","?","than"]))
            ])),
            ('bag-of-words', Pipeline([
                ('tfidf', TfidfVectorizer())
            ])),
            ('bag-of-words-first', Pipeline([
                ('before-first', BeforeFirstObject('OBJECT_A')),
                ('tfidf-2', TfidfVectorizer())
            ]))
        ])),
        ('model', LogisticRegression())
    ])
    model = pipeline.fit(train['text'].values, train['label'].values)
    
    predictions = model.predict(test['text'].values)

    print(classification_report(test['label'], predictions))
    print(confusion_matrix(test['label'], predictions))


if __name__ == '__main__':
    run_pipeline(LogisticRegression())
