import codecs
import re
import traceback
from collections import defaultdict

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
from sklearn.metrics import f1_score

from tools.features.LengthOfSentence import LengthAnalyzer
from tools.features.SentenceParts import BeforeAfterWord
from xgboost import XGBClassifier
from datetime import datetime
from sklearn.utils import  shuffle

from tools.features.WordEmbedding import Glove100, Glove300

K_BEST_K = 'all'


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


TRAIN = build_data_frame('data/train_80.csv')
TEST = build_data_frame('data/test_20.csv')
DATA = shuffle( pd.concat([TRAIN, TEST]), random_state=42)
STOPWORDS = stopwords.words('english')

k_fold = KFold(n_splits=5)
splits = list(k_fold.split(DATA['text'], DATA['label']))

LABELS = ['A_GREATER_B', 'A_LESSER_B', 'NO_COMP']


def run_pipeline(prefix, model, save_feat=False):
    try:
        print('\n\n=== {} ({})==='.format(type(model).__name__, prefix))
        for train_index, test_index in splits:
            train = DATA.iloc[train_index]
            test = DATA.iloc[test_index]
            embedding = WordEmbedding()
            k_best = SelectKBest(f_classif, k=K_BEST_K)

            union = FeatureUnion(
                [
                    # ('between-a-b', Pipeline([
                    #     ('extract', BetweenWords("OBJECT_A", "OBJECT_B", first_occ_a=True, first_occ_b=True)),
                    #     ('bow', CountVectorizer()),
                    # ])),
                    # ('context-a', Pipeline([
                    #     ('extract', BeforeAfterWord('OBJECT_A', before=True, first_occ=True)),
                    #     ('bow', CountVectorizer())
                    # ])),
                    # ('context-b', Pipeline([
                    #     ('extract', BeforeAfterWord('OBJECT_B', before=False, first_occ=True)),
                    #     ('bow', CountVectorizer()),
                    # ])),
                    ('embedding', Pipeline([
                        ('w2v', embedding)
                    ])),
                    ('tf-idf', Pipeline([
                        ('tf-idf', TfidfVectorizer())
                    ])),
                    # ('length', Pipeline([
                    #     ('length-whole', LengthAnalyzer())
                    # ]))

                ])
            pipeline = Pipeline([
                ('features', union),
                #    ('k-best', k_best),
                ('model', model),
            ])

            predictions = pipeline.fit(train['text'].values, train['label'].values).predict(test['text'].values)
            # print("###############")
            # print(cv.best_estimator_)
            # print(cv.best_params_)

            report = classification_report(test['label'].values, predictions, labels=LABELS)
            f1 = '{:02.2f}'.format(f1_score(test['label'].values, predictions, labels=LABELS, average='weighted') * 100)
            matrix = confusion_matrix(test['label'].values, predictions, labels=LABELS)

            print(report, matrix)

            if save_feat and False:
                feature_names = feat_names("BETW", union, 0) + feat_names("CTX_A", union, 1) + feat_names("CTX_B",
                                                                                                          union, 2)
                save_features(feature_names, k_best, model, summary(
                    [union.transformer_list[0][1].named_steps['extract'],
                     union.transformer_list[1][1].named_steps['extract'],
                     union.transformer_list[2][1].named_steps['extract']], k_best, model, report, matrix),
                              './feature_checks/{}-{}-{}.csv'.format(f1, prefix, type(model).__name__))
        print("#" * 100)
    except Exception as e:
        print('{} {} failed'.format(type(model).__name__, prefix))
        print(e)


def grid_search(pipeline):
    return GridSearchCV(pipeline, verbose=1, n_jobs=10, param_grid={
        'features__between-a-b__extract__first_occ_a': [True, False],
        'features__between-a-b__extract__first_occ_b': [True, False],
        'features__context-a__extract__before': [True, False],
        'features__context-a__extract__first_occ': [True, False],
        'features__context-b__extract__before': [True, False],
        'features__context-b__extract__first_occ': [True, False],
        # 'features__transformer_weights': [
        #     {'between-a-b': 1, 'context-a': 1, 'context-b': 1, 'embedding': 1, "tf-idf": 1},
        #     {'between-a-b': 1, 'context-a': 1, 'context-b': 1, 'embedding': 1, "tf-idf": 0},
        #     {'between-a-b': 1, 'context-a': 1, 'context-b': 1, 'embedding': 0, "tf-idf": 1},
        #     {'between-a-b': 1, 'context-a': 1, 'context-b': 1, 'embedding': 0, "tf-idf": 0},
        #     {'between-a-b': 1, 'context-a': 1, 'context-b': 1, 'embedding': 0.5, "tf-idf": 0.5},
        #
        #
        # ]
    })


def feat_names(prefix, union, index):
    return [prefix + '_' + feat for feat in union.transformer_list[index][1].named_steps['bow'].get_feature_names()]


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
       PassiveAggressiveClassifier()
        # XGBClassifier(max_depth=6),
    ]

    for alg in algo:
        run_pipeline('', alg, save_feat=False)
