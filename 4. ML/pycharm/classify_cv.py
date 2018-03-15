import random

import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from features.ngram_feature import NGramFeature
from infersent.infersent_feature import initialize_infersent, InfersentFeature
from transformers.data_extraction import ExtractMiddlePart
from transformers.n_gram_transformers import NGramTransformer
from util.data_utils import load_data
from util.misc_utils import get_logger
from util.ngram_utils import get_all_ngrams

LABEL = 'most_frequent_class'

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

np.random.seed(1337)
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


def run_classification(data):
    params = [
        # default model
        {'max_depth': [6],
         'min_child_weight': [1],
         'gamma': [0],
         'subsample': [1]},
        # more complex
        {'max_depth': [12],
         'min_child_weight': [0]
         },
        # less complex
        {'max_depth': [2]},
        # more regulized
        {'alpha': [0.5],
         'lambda': [2]
         }
    ]
    logger.info(params)
    cv = GridSearchCV(XGBClassifier(), param_grid=params, cv=5, verbose=5,
                      scoring=make_scorer(f1_score, average='weighted'))

    raw_text = ExtractMiddlePart().transform(data)
    infersent_model = initialize_infersent(raw_text)
    pipeline = make_pipeline(ExtractMiddlePart(), InfersentFeature(infersent_model), cv)
    print(pipeline.get_params().keys())
    fitted = pipeline.fit(data, data[LABEL].values)
    logger.info(cv.best_params_)
    logger.info(cv.best_score_)


logger.info('# THREE CLASSES')
_data = load_data('data.csv', binary=False)
run_classification(_data)
logger.info('\n\n--------------------------------------------\n\n')
logger.info('# BINARY CLASSES')
_data_bin = load_data('data.csv', binary=True)

# run_classification(_data_bin, ['ARG', 'NONE'])
