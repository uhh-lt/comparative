from pprint import pformat
import pandas as pd
import spacy
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from util.classification_report_util import get_std_derivations, latex_classification_report, get_avg_fold
from transformers.data_extraction import ExtractRawSentence
from util.data_utils import load_data, k_folds
from util.misc_utils import get_logger

"""
This script was used to check the performance of different classifiers on the task using a Bag-Of-Words feature.
"""
nlp = spacy.load('en_core_web_lg')

logger = get_logger('feat_tests_1_neu')

LABEL = 'most_frequent_label'
data = load_data('data.csv')

best_per_feat = []

class_result = pd.DataFrame(columns=['classifier', 'f1', 'precision', 'recall'])

classifiers = [('XGBoost', XGBClassifier(n_jobs=8, n_estimators=1000)), ('Logistic Regression', LogisticRegression()),
               ('AdaBoost', AdaBoostClassifier()), ('SVM (linear)', LinearSVC()),
               ('Decision Tree', DecisionTreeClassifier()),
               ('SGD Classifier', SGDClassifier()), ('Random Forest', RandomForestClassifier()), ('Extra Trees', ExtraTreesClassifier()), ('k-Neighbors', KNeighborsClassifier()),
               ('SVM (radial basis function)', SVC(kernel='rbf')), ('SVM (polynomial)', SVC(kernel='poly')), ('SVM (sigmoid)', SVC(kernel='sigmoid')),
               ('Multinomial NB', MultinomialNB()), ('Majority Class Baseline', DummyClassifier(strategy='most_frequent'))
               ]

idx = 0
for name, classifier in classifiers:
    logger.info(classifier)
    folds_results = []

    try:
        for train, test in k_folds(5, data, random_state=42):
            pipeline = make_pipeline(FeatureUnion(
                [('unigram counts binary all', make_pipeline(ExtractRawSentence(), CountVectorizer(binary=True)))]),
                classifier)

            fitted = pipeline.fit(train, train[LABEL].values)
            predicted = fitted.predict(test)

            class_result.loc[idx] = [name, f1_score(test[LABEL].values, predicted, average='weighted'), precision_score(test[LABEL].values, predicted, average='weighted'),
                                     recall_score(test[LABEL].values, predicted, average='weighted')]
            idx += 1

            folds_results.append((test[LABEL].values, predicted))
            logger.info(
                classification_report(test[LABEL].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
        der = get_std_derivations(folds_results, ['BETTER', 'WORSE', 'NONE'])
        best = get_avg_fold(folds_results)
        best_per_feat.append((f1_score(best[0], best[1], average='weighted'), classifier))
        #   print(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
        logger.info(
            latex_classification_report(best[0], best[1], derivations=der, labels=['BETTER', 'WORSE', 'NONE'],
                                        caption=''))
    except Exception as ex:
        logger.error(ex)
    logger.info("\n\n=================\n\n")

logger.info(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
class_result.to_csv('graphics/data/classifer_test.csv')
