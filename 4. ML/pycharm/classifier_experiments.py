import itertools

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from classification_report_util import get_std_derivations, get_best_fold, latex_classification_report
from features.contains_features import ContainsPos
from features.mean_embedding_feature import MeanWordEmbedding
from infersent.infersent_feature import InfersentFeature, initialize_infersent
from transformers.data_extraction import ExtractRawSentence, ExtractMiddlePart, ExtractFirstPart, ExtractLastPart
from util.data_utils import load_data, k_folds
from util.misc_utils import get_logger
from pprint import pformat

nlp = spacy.load('en')

logger = get_logger('feat_tests_1')

LABEL = 'most_frequent_label'
data = load_data('data.csv')

best_per_feat = []

classifiers = [XGBClassifier(n_jobs=8, n_estimators=100), LogisticRegression(), AdaBoostClassifier(), LinearSVC(),
               DecisionTreeClassifier(),
               SGDClassifier(), RandomForestClassifier(), ExtraTreesClassifier(), KNeighborsClassifier(),
               RadiusNeighborsClassifier(), SVC(class_weight='balanced'),
               SVC(kernel='rbf'), SVC(kernel='poly'), SVC(kernel='sigmoid'), GaussianNB(), MultinomialNB()]

for classifier in classifiers:
    logger.info(classifier)
    folds_results = []
    try:
        for train, test in k_folds(5, data, random_state=1337):
            pipeline = make_pipeline(FeatureUnion(
                [('unigram counts binary all', make_pipeline(ExtractRawSentence(), CountVectorizer(binary=True)))]),
                                     classifier)

            fitted = pipeline.fit(train, train[LABEL].values)
            predicted = fitted.predict(test)
            folds_results.append((test[LABEL].values, predicted))
            logger.info(
                classification_report(test[LABEL].values, predicted, labels=['BETTER', 'WORSE', 'NONE'], digits=2))
        der = get_std_derivations(folds_results, ['BETTER', 'WORSE', 'NONE'])
        best = get_best_fold(folds_results)
        best_per_feat.append((f1_score(best[0], best[1], average='weighted'), classifier))
     #   print(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
        logger.info(
            latex_classification_report(best[0], best[1], derivations=der, labels=['BETTER', 'WORSE', 'NONE'],
                                        caption=''))
    except Exception as ex:
        logger.error(ex)
    logger.info("\n\n=================\n\n")

logger.info(pformat(sorted(best_per_feat, key=lambda k: k[0], reverse=True)))
