from itertools import combinations

from bs4 import BeautifulSoup
from pandas import DataFrame as df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import subprocess

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


def get_score(test, predictions, labels):
    print(classification_report(test['label'].values, predictions, labels=labels))
    print(confusion_matrix(test['label'].values, predictions, labels=labels))
    return f1_score(test['label'].values, predictions, labels=labels, average='weighted')


def linear_svc(features, splits, labels):
    print('********* SVC *********')
    pipeline = make_pipeline(FeatureUnion(features), LinearSVC())

    for split in splits:
        train, test = split
        fitted = pipeline.fit(train,
                              train['label'].values)
        predictions = fitted.predict(test)

        get_score(test, predictions, labels)


def decision_tree(features, splits, labels):
    print('********* DecisionTree *********')

    for i, split in enumerate(list(splits)[:1]):
        classifier = DecisionTreeClassifier(max_depth=10, max_features= 'auto')
        union = FeatureUnion(features)
        pipeline = make_pipeline(union, classifier)

        param_grid = {
            'max_depth': [None, 10, 100], 'max_features': [None, 'auto', 'log2'], 'max_leaf_nodes': [None, 5, 10]
        }

        # pipeline = GridSearchCV(pipeline, param_grid={}, cv=splits)

        train, test = split
        fitted = pipeline.fit(train,
                              train['label'].values)

        predictions = fitted.predict(test)

        export_graphviz(
            classifier,
            out_file='tree-{}.dot'.format(i),
            feature_names=get_feature_names(fitted),
            class_names=_labels,
            rounded=True,
            filled=True
        )
        subprocess.call(['dot', '-Tpdf', 'tree-{}.dot'.format(i), '-o' 'tree-{}.pdf'.format(i)])

        get_score(test, predictions, labels)


def get_feature_names(pipeline):
    lst = pipeline.named_steps['featureunion'].transformer_list
    names = []
    for t in lst:
        names += t[1].steps[-1][1].get_feature_names()
    print(names)
    return names


def logistic_regression(features, splits, labels):
    print('********* LogisticRegression *********')
    pipeline = make_pipeline(FeatureUnion(features), LogisticRegression())

    for split in splits:
        train, test = split
        fitted = pipeline.fit(train,
                              train['label'].values)
        predictions = fitted.predict(test)

        get_score(test, predictions, labels)


def sgd(features, splits, labels):
    print('********* SGD *********')
    pipeline = make_pipeline(FeatureUnion(features), SGDClassifier())

    for split in splits:
        train, test = split
        fitted = pipeline.fit(train,
                              train['label'].values)
        predictions = fitted.predict(test)

        get_score(test, predictions, labels)


if __name__ == '__main__':
    _labels = ['BETTER', 'WORSE', 'OTHER', 'NONE']
    _data = load_data('train-data.csv')
    unigrams, bigrams, trigrams = [], [], []  # setup_n_grams(_data)
    print('Build n-grams')
    ## 0.704
    best_so_far = [

        ('better-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER))),  # 0.613
        # ('tfidf', make_pipeline(ExtractRawSentence(), TfidfVectorizer())),  # 0.621
        ('ngram', make_pipeline(ExtractRawSentence(), NGramTransformer(n=1), NGramFeature(unigrams))),  # 0.628
        ('bw-m', make_pipeline(ExtractMiddlePart(), ContainsWord(CUE_WORDS_BETTER + CUE_WORDS_WORSE))),  # 0.645
        #  ('tf-m', make_pipeline(ExtractMiddlePart(), TfidfVectorizer())),  # 0.671
        ('bi-m', make_pipeline(ExtractMiddlePart(), NGramTransformer(n=2), NGramFeature(bigrams))),  # 0.671
        ('uni-m', make_pipeline(ExtractMiddlePart(), NGramTransformer(n=1), NGramFeature(unigrams))),  # 0.678

    ]
    print('Build features')
    splits = list(split_data(3, _data))
    decision_tree(best_so_far, splits, _labels)
    # linear_svc(best_so_far)
    # logistic_regression(best_so_far)
    # sgd(best_so_far)
