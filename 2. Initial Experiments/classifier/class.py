import re

from pandas import DataFrame
from tools import StopwordRemover, LengthAnalyzer, WordOccurence, BeforeAfterWord, BetweenWords, WordPos, ObjectContext, \
    NumWordOccurence, WordEmbeddingVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import  LinearSVC
from pprint import pprint
from random import shuffle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from gensim.models import Word2Vec, KeyedVectors


def build_data_frame(file_name):
    rows = []
    with open(file_name, 'r') as data:
        for line in data:
            try:
                text, label = line.rsplit(';')
                rows.append({'text': text.strip(), 'label': label.strip().replace('?', '')})
            except Exception as e:
                pass
    data_frame = DataFrame(rows)

    return data_frame


# nltk.download('stopwords')
# nltk.download('punkt')

DATA = build_data_frame('data/binary.csv')

LABELS = ['COMP', 'NO_COMP']


def run_pipeline(model):
    train, test = train_test_split(DATA, test_size=0.3, shuffle=True)

    #  kf = KFold(n_splits=3, shuffle=True)
    f1 = []
    # for train_idx, test_idx in kf.split(DATA['text'].values, DATA['label'].values):
    # train = DATA.iloc[train_idx]

    #  test = DATA.iloc[test_idx]

  #  w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    union = FeatureUnion(
        [

            # ('between-a-b', Pipeline([
            #     ('feat', BetweenWords('OBJECT_A', 'OBJECT_B')),
            #     ('bow', CountVectorizer()),
            #     ('trans', TfidfTransformer())
            # ])),
            # #
            # ('before-first-object', Pipeline([
            #     ('feat', ObjectContext(True)),
            #     ('bow',  CountVectorizer()),
            #     ('trans', TfidfTransformer())
            # ])),
            # ('after-second-object', Pipeline([
            #     ('feat', ObjectContext(False)),
            #     ('bow',  CountVectorizer()),
            #     ('trans', TfidfTransformer())
            # ])),

            #
            # ('bow-before-obj-a', Pipeline([
            #     ('before-obj-a', BeforeAfterWord('OBJECT_A', True)),
            #     ('tfidf', CountVectorizer())
            # ])),
            # # ('bow-before-obj-b', Pipeline([
            # #     ('before-obj-b', BeforeAfterWord('OBJECT_B', True)),
            # #     ('tfidf', CountVectorizer())
            # # ])),
            # # ('bow-after-obj-a', Pipeline([
            # #     ('before-obj-b', BeforeAfterWord('OBJECT_A', False)),
            # #     ('tfidf', CountVectorizer())
            # # ])),
            # # ('bow-after-obj-b', Pipeline([
            # #     ('after-obj-b', BeforeAfterWord('OBJECT_B', False)),
            # #     ('tfidf', CountVectorizer())
            # # ])),
            ('bag-of-words-whole', Pipeline([
                ('count', CountVectorizer()),
            ])),
        ])
    pipeline = Pipeline([
        ('features', union),
        ('model', model)
    ])

    modela = pipeline.fit(train['text'].values, train['label'].values)
    predictions = modela.predict(test['text'].values)

    print(classification_report(test['label'].values, predictions, labels=LABELS))
    print(confusion_matrix(test['label'].values, predictions, labels=LABELS))


if __name__ == '__main__':
    print("== Logistic Regression ==")
    run_pipeline(LogisticRegression())
    print("\n\n== Linear SVC ==")
    run_pipeline(LinearSVC())
    print("\n\n== SGDClassifier ==")
    run_pipeline(SGDClassifier())
    print("\n\n== MultinomialNB ==")
    run_pipeline(MultinomialNB())
    print("\n\n== PassiveAggressiveClassifier ==")
    run_pipeline(PassiveAggressiveClassifier())
