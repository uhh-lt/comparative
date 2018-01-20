from itertools import combinations

from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from features.ngram_feature import NGramFeature
from transformers.data_extraction import *
from transformers.n_gram_transformers import NGramTransformer
from util.feature_builder import *
from util.ngram import get_all_ngrams

CUE_WORDS_WORSE = ["worse", "harder", "slower", "poorly", "uglier", "poorer", "lousy", "nastier", "inferior",
                   "mediocre"]

CUE_WORDS_BETTER = ["better", "easier", "faster", "nicer", "wiser", "cooler", "decent", "safer", "superior", "solid",
                    "teriffic"]


def load_data(file_name, min_confidence=0.6, binary=False):
    frame = df.from_csv(path='data/' + file_name)
    frame = frame[frame['label:confidence'] >= min_confidence]
    frame['raw_text'] = frame.apply(
        lambda row: BeautifulSoup(row['text_html'], "lxml").text.replace(':[OBJECT_A]', '').replace(':[OBJECT_B]', ''),
        axis=1)
    if binary:
        frame['label'] = frame.apply(lambda row: row['label'] if row['label'] == 'NONE' else 'ARG', axis=1)
    return shuffle(frame)


def split_data(splits, data):
    """create splits for k fold validation"""
    k_fold = StratifiedKFold(n_splits=splits, random_state=42)
    for train_index, test_index in k_fold.split(data,
                                                data['label']):
        yield data.iloc[train_index], data.iloc[test_index]


def setup_n_grams(d):
    return get_all_ngrams(d['raw_text'].values, n=1), get_all_ngrams(d['raw_text'].values, n=2), get_all_ngrams(
        d['raw_text'].values, n=3)


def build_features():
    feat = best_so_far

    combis = list(combinations(feat, len(feat)))
    print('{} feature combinations'.format(len(combis)))
    return combis


def run_pipeline(estimator, feature_union, train, test, labels):
    pipeline = Pipeline([('features', feature_union), ('estimator',
                                                       estimator)])

    fitted = pipeline.fit(train,
                          train['label'].values)

    predictions = fitted.predict(test)

    report = classification_report(test['label'].values, predictions, labels=labels)
    conf = confusion_matrix(test['label'].values, predictions, labels=labels)
    f1 = f1_score(test['label'].values, predictions, labels=labels, average='weighted')
    return f1, str(report), str(conf)


def run_estimator(estimator, data, label, features):
    key = str(type(estimator))
    print('Start {}'.format(key))
    f1_sum = 0
    for train, test in split_data(3, data):
        f1, report, conf = run_pipeline(estimator, FeatureUnion(features), train, test, label)
        print(report)
        print(conf)
        f1_sum += f1
    print('\nAverage F1 Score {}\n==========================\n\n'.format(f1_sum / 3))
    return f1_sum / 3


if __name__ == '__main__':
    _labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
    _data = load_data('train-data.csv')
    unigrams, bigrams, trigrams = setup_n_grams(_data)
    ## 0.704
    best_so_far = [

        ('better-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER))),  # 0.613
        ('vsm', make_pipeline(ExtractRawSentence(), TfidfVectorizer())),  # 0.621
        ('uni', make_pipeline(ExtractRawSentence(), NGramTransformer(n=1), NGramFeature(unigrams))),  # 0.628
        ('bw-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),  # 0.645
        ('vsm-m', make_pipeline(ExtractMiddlePart(), TfidfVectorizer())),  # 0.671
        ('bigram-m', make_pipeline(ExtractMiddlePart(), NGramTransformer(n=2), NGramFeature(bigrams))),  # 0.671
        ('uni-m', make_pipeline(ExtractMiddlePart(), NGramTransformer(n=1), NGramFeature(unigrams))),  # 0.678

    ]

    f1 = 0
    estimators = [LogisticRegression(), DecisionTreeClassifier(), LinearSVC(), SGDClassifier()]

    for estimator in estimators:
        f1 += run_estimator(estimator, _data, _labels, best_so_far)
        print('\n**************************************************\n>>>>>> {}\n\n'.format(f1 / len(estimators)))
        f1 = 0
