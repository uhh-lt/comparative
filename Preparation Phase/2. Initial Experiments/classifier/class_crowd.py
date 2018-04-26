import codecs
import re
import traceback
from collections import defaultdict

import spacy

from pandas import DataFrame
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from tools import WordEmbedding, BetweenWords
from sklearn.metrics import f1_score, make_scorer

from tools.features.LengthOfSentence import LengthAnalyzer
from tools.features.SentenceParts import BeforeAfterWord
from xgboost import XGBClassifier
from datetime import datetime
from sklearn.utils import shuffle

# from tools.features.WordEmbedding import Glove100, Glove300
from tools.features.WordEmbedding import SpacyEmbedding
from tools.features.spacy_feat import POSCount, NERCount, POSSequence

K_BEST_K = 'all'


def build_data_frame(file_name):
    return DataFrame.from_csv(path=file_name)


nlp = spacy.load('en')
#TRAIN = build_data_frame('data/train_80.csv')
#TEST = build_data_frame('data/test_20.csv')
DATA = shuffle(build_data_frame('data/200-cw.csv'), random_state=42)

STOPWORDS = stopwords.words('english')

k_fold = StratifiedKFold(n_splits=3)
splits = list(k_fold.split(DATA['sentence'], DATA['label']))

LABELS = ['BETTER', 'WORSE', 'UNCLEAR',  'NO_COMP']



# embedding = WordEmbedding()


def run_pipeline(prefix, model, save_feat=False):
    print('\n\n=== {} ({})==='.format(type(model).__name__, prefix))
    f1s = []
    #for train_index, test_index in splits:
    #train = DATA.iloc[train_index]
    #test = DATA.iloc[test_index]
    train = DATA[:150]
    test = DATA[150:]
    k_best = SelectKBest(f_classif, k=K_BEST_K)

    vectorizer = TfidfVectorizer()
    union = FeatureUnion(
        [
            # ('between-a-b', Pipeline([
            #     ('extract', BetweenWords("OBJECT_A", "OBJECT_B", first_occ_a=True, first_occ_b=True)),
            #     ('bow', CountVectorizer()),
            # ])),
            # ('context-a', Pipeline([
            #     ('extract', BeforeAfterWord('OBJECT_A', before=False, first_occ=False)),
            #     ('bow', CountVectorizer())
            # ])),
            # ('context-b', Pipeline([
            #     ('extract', BeforeAfterWord('OBJECT_B', before=False, first_occ=True)),
            #     ('bow', CountVectorizer()),
            # ])),
            ('embedding', Pipeline([
                ('w2v', SpacyEmbedding(nlp))
            ])),
            # ('tf-idf', Pipeline([
            #     ('tf-idf', CountVectorizer())
            # ])),
            # ('noun-count', Pipeline([
            #     ('noun', POSCount(nlp, spacy.parts_of_speech.NOUN))
            # ])),
            # ('adj-count', Pipeline([
            #     ('adj', POSCount(nlp, spacy.parts_of_speech.ADJ))
            # ])),
            # ('ner-count', Pipeline([
            #     ('ner', NERCount(nlp))
            # ])),
            # ('pos-seq', Pipeline([
            #     ('seq', POSSequence(nlp)),
            #     ('bow', CountVectorizer())
            # ])),
            #
            # ('length', Pipeline([
            #     ('length-whole', LengthAnalyzer())
            # ]))

        ])
    pipeline = Pipeline([
        ('features', union),
        # ('k-best', k_best),
        ('model', model),
    ])
    # cv = grid_search(pipeline)
    fit = pipeline.fit(train['sentence'].values, train['label'].values)

    predictions = fit.predict(test['sentence'].values)

    f1s.append(f1_score(test['label'], predictions, average='weighted'))

    # save_prob(pipeline, train, test)
    # print("###############")
    # print(cv.best_estimator_)
    # print("###############")
    # print(cv.best_params_)
    # print("###############")

    report = classification_report(test['label'].values, predictions, labels=LABELS)
    # report = classification_report(test['label'].values,
    #                                cv.best_estimator_.fit(train['text'].values, train['label'].values).predict(
    #                                    test['text'].values), labels=LABELS)
    f1 = '{:02.2f}'.format(f1_score(test['label'].values, predictions, labels=LABELS, average='weighted') * 100)
    matrix = confusion_matrix(test['label'].values, predictions, labels=LABELS)

    print(report, matrix)

    if save_feat:
        feature_names = feat_names("BETW", union, 0) + feat_names("CTX_A", union, 1) + feat_names("CTX_B",
                                                                                                  union,
                                                                                                  2) + vectorizer.get_feature_names()
        save_features(feature_names, k_best, model, summary(
            [union.transformer_list[0][1].named_steps['extract'],
             union.transformer_list[1][1].named_steps['extract'],
             union.transformer_list[2][1].named_steps['extract']], k_best, model, report, matrix),
                      './{}-{}-{}.csv'.format(f1, prefix, type(model).__name__))

    # print("#######################################")
    # print(np.mean(np.array(f1s)))


def grid_search(pipeline):
    scorer = make_scorer(f1_score, average='weighted')
    return GridSearchCV(pipeline, verbose=1, n_jobs=16, param_grid={
        # 'features__between-a-b__extract__first_occ_a': [True, False],
        # 'features__between-a-b__extract__first_occ_b': [True, False],
        'features__context-a__extract__before': [True, False],
        'features__context-a__extract__first_occ': [True, False],
        # 'features__context-b__extract__before': [True, False],
        # 'features__context-b__extract__first_occ': [True, False],
        # 'features__transformer_weights': [
        # {'between-a-b': 1, 'context-a': 0, 'context-b': 0, 'embedding': 1, "tf-idf": 1},
        # {'between-a-b': 1, 'context-a': 1, 'context-b': 1, 'embedding': 1, "tf-idf": 1},
        # {'between-a-b': 1, 'context-a': 1, 'context-b': 1, 'embedding': 1, "tf-idf": 0.6},
        # {'between-a-b': 2, 'context-a': 1, 'context-b': 1, 'embedding': 1, "tf-idf": 1},
        # {'between-a-b': 3, 'context-a': 1, 'context-b': 1, 'embedding': 3, "tf-idf": 3},
        # {'between-a-b': 2, 'context-a': 1, 'context-b': 1, 'embedding': 2, "tf-idf": 1},
        # {'between-a-b': 2, 'context-a': 1, 'context-b': 1, 'embedding': 1, "tf-idf": 2},
        # {'between-a-b': 1, 'context-a': 1, 'context-b': 1, 'embedding': 0.5, "tf-idf": 0.5},
        # {'between-a-b': 1, 'context-a': 0.5, 'context-b': 0.5, 'embedding': 1, "tf-idf": 1},
        # {'between-a-b': 1, 'context-a': 0.5, 'context-b': 0.5, 'embedding': 0.5, "tf-idf": 1},
        # {'between-a-b': 0.5, 'context-a': 0.5, 'context-b': 0.5, 'embedding': 0.5, "tf-idf": 1},

        # ]
    })


def feat_names(prefix, union, index):
    return [prefix + '_' + feat for feat in union.transformer_list[index][1].named_steps['bow'].get_feature_names()]


def save_prob(pipeline, train, test):
    df = DataFrame(
        np.array(pipeline.fit(train['text'].values, train['label'].values).predict_proba(test['text'].values)) * 100,
        columns=LABELS)

    df['Sentence'] = test['text'].values
    df.to_csv(path_or_buf='./data/conf.csv', index=False, float_format='%.2f')


def summary(*objs):
    desc = []
    for o in objs:
        desc.append('\n{}\n'.format(o))
    return '\n'.join(desc) + '\n\n'


def save_features(feature_names, k_best, model, sum, output_fpath):
    support = k_best.get_support()
    feature_names = np.asarray(feature_names)[support]

    features = defaultdict(list)
    for k, c in enumerate(model.classes_):
        if k > len(model.coef_) - 1:
            continue  # when we have two classes (binary classification) there will be just one feature vector
        for i, fn in enumerate(feature_names):
            features[c].append((model.coef_[k][i], fn))

    with codecs.open(output_fpath, "w", "utf-8") as output_file:
        print(sum + "\n", file=output_file)
        for c in features:
            print('\n==== {} ====\n'.format(c), file=output_file)
            for f in sorted(features[c], reverse=True)[0:50]:
                try:
                    print("%s;%s;%f" % (c, f[1], f[0]), file=output_file)
                except:
                    print("Bad feature:", c, f)
                    print(traceback.format_exc())


if __name__ == '__main__':

    algo = [
        LogisticRegression(),
        LinearSVC(),
        SGDClassifier(),
        Perceptron(),
        RandomForestClassifier(),
        MLPClassifier(),
        PassiveAggressiveClassifier(),
        XGBClassifier(max_depth=6),
    ]

    for alg in algo:
        run_pipeline('', alg, save_feat=False)
