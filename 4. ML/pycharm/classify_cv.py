from nltk import NaiveBayesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from features.contains_features import ContainsPos
from features.count_features import PunctuationCount, NamedEntitiesByCategory, NounChunkCount
from features.mean_embedding_feature import MeanWordEmbedding
from features.misc_features import PositionOfObjects
from features.ngram_feature import NGramFeature
from infersent.infersent_feature import initialize_infersent, InfersentFeature
from transformers.data_extraction import ExtractRawSentence, ExtractMiddlePart
from transformers.n_gram_transformers import NGramTransformer
from util.data_utils import load_data, k_folds
from util.misc_utils import latex_table, get_logger
from util.ngram_utils import get_all_ngrams
from collections import OrderedDict
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier
import datetime

classifier_pattern = """
----------------------
- {}
----------------------"""
feature_name_pattern = """
========================================================================================

======================
- {}
======================"""

logger = get_logger('xgb_random')

classifiers = [XGBClassifier()]
# classifiers = [RidgeClassifier(), XGBClassifier(), LinearSVC(), SVC(), SGDClassifier(), GaussianNB(),
#               KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
#               LogisticRegression(), ExtraTreesClassifier()]

n_gram_cache = {}

import numpy as np

np.random.seed(1337)
import random

random.seed(1337)


def n_gram_pipeline(n, extractor, processing=None):
    def _n_gram_pipeline(train):
        ngram_base = extractor(processing=processing).transform(train)
        unigrams = get_all_ngrams(ngram_base, n)
        return [extractor(processing=processing), NGramTransformer(n), NGramFeature(unigrams, n=n)]

    return _n_gram_pipeline


def infersent_pipeline(extractor, processing=None):
    def _infersent_pipeline(train):
        raw_text = extractor(processing=processing).transform(train)
        infersent_model = initialize_infersent(raw_text)
        return [extractor(processing=processing), InfersentFeature(infersent_model)]

    return _infersent_pipeline


feature_builder = [
    ('Sentence Embedding MP', infersent_pipeline(ExtractMiddlePart)),

]


def run_classification(data, labels):
    by_score = []

    params = {

        'n_estimators': [200, 300, 400],
        'max_depth': [6, 12, 24],
        'learning_rate': [0.01, 0.3],

        'reg_lambda': [0, 1, 5, 100],
        'reg_alpha': [0, 1, 5, 100]
    }
    logger.info(params)
    cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, cv=5, verbose=5,
                            scoring=make_scorer(f1_score, average='weighted'))

    raw_text = ExtractMiddlePart().transform(data)
    infersent_model = initialize_infersent(raw_text)
    pipeline = make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model), cv)
    print(pipeline.get_params().keys())
    fitted = pipeline.fit(data, data['label'].values)
    logger.info(cv.best_params_)
    logger.info(cv.best_score_)


logger.info('# THREE CLASSES')
_data = load_data('data.csv', min_confidence=0, binary=False)
run_classification(_data, ['BETTER', 'WORSE', 'NONE'])
logger.info('\n\n--------------------------------------------\n\n')
logger.info('# BINARY CLASSES')
_data_bin = load_data('data.csv', min_confidence=0, binary=True)

# run_classification(_data_bin, ['ARG', 'NONE'])
